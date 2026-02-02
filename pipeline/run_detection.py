#!/usr/bin/env python3
"""
SF Security Camera - Fall and Fight Detection using GStreamer
Uses NVIDIA DeepStream elements via GStreamer Python bindings
"""

import sys
import os
import json
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib, GstVideo

# Configuration
RECORDINGS_DIR = Path("/app/recordings")
LOGS_DIR = Path("/app/logs")

class IncidentTracker:
    """Track incidents and write to log files"""
    def __init__(self):
        self.incidents = []
        self.lock = threading.Lock()
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def log_incident(self, incident_type, description, metadata=None):
        incident = {
            'timestamp': datetime.now().isoformat(),
            'type': incident_type,
            'description': description,
            'metadata': metadata or {}
        }

        with self.lock:
            self.incidents.append(incident)
            log_file = LOGS_DIR / f"incidents_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(incident) + '\n')

        print(f"\n{'='*60}")
        print(f"[INCIDENT] {incident_type} DETECTED!")
        print(f"Time: {incident['timestamp']}")
        print(f"Description: {description}")
        print(f"{'='*60}\n")

tracker = IncidentTracker()


def create_pipeline(source_uri):
    """Create GStreamer pipeline with DeepStream elements"""
    Gst.init(None)

    # Determine source type
    if source_uri.startswith('rtsp://'):
        source_str = f'rtspsrc location="{source_uri}" latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder'
    elif source_uri.startswith('/dev/video'):
        source_str = f'v4l2src device="{source_uri}" ! video/x-raw,width=1280,height=720 ! nvvideoconvert'
    else:
        source_str = f'filesrc location="{source_uri}" ! qtdemux ! h264parse ! nvv4l2decoder'

    # Pipeline with DeepStream inference
    pipeline_str = f'''
        {source_str} !
        m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 !
        nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt !
        nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
            ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml
            tracker-width=640 tracker-height=480 !
        nvvideoconvert !
        nvdsosd !
        tee name=t
        t. ! queue ! nvvideoconvert ! nveglglessink sync=0
        t. ! queue ! nvvideoconvert ! nvv4l2h264enc bitrate=4000000 ! h264parse ! mp4mux ! filesink location=/app/recordings/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4
    '''

    print(f"Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return None


def create_simple_pipeline(source_uri):
    """Create a simpler pipeline for testing"""
    Gst.init(None)

    if source_uri.startswith('rtsp://'):
        pipeline_str = f'''
            rtspsrc location="{source_uri}" latency=100 !
            rtph264depay ! h264parse ! nvv4l2decoder !
            nvvideoconvert ! nveglglessink sync=0
        '''
    else:
        pipeline_str = f'''
            filesrc location="{source_uri}" !
            qtdemux ! h264parse ! nvv4l2decoder !
            nvvideoconvert ! nveglglessink sync=0
        '''

    print(f"Simple pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return None


def on_message(bus, message, loop):
    """Handle pipeline messages"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    return True


def main():
    print("="*60)
    print("SF Security Camera - Fall & Fight Detection")
    print("="*60)

    # Get source from command line
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4'

    print(f"Source: {source}")
    print(f"Recordings: {RECORDINGS_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print("="*60)

    # Try full pipeline first, fall back to simple
    pipeline = create_pipeline(source)
    if not pipeline:
        print("Falling back to simple pipeline...")
        pipeline = create_simple_pipeline(source)
        if not pipeline:
            print("Failed to create pipeline")
            sys.exit(1)

    # Set up message handling
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Start playing
    print("\nStarting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        sys.exit(1)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping...")

    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")


if __name__ == '__main__':
    main()
