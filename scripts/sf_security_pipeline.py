#!/usr/bin/env python3
"""
SF Security Camera - Fall and Fight Detection Pipeline
Uses NVIDIA DeepStream for real-time video analytics
"""

import sys
import os
import gi
import time
import json
import threading
from datetime import datetime
from collections import defaultdict
from pathlib import Path

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

import pyds

# Configuration
RECORDINGS_DIR = Path("/app/recordings")
LOGS_DIR = Path("/app/logs")
INCIDENT_BUFFER_SECONDS = 10
FALL_CONFIDENCE_THRESHOLD = 0.7
FIGHT_CONFIDENCE_THRESHOLD = 0.6

# Tracking state for action detection
class PersonTracker:
    def __init__(self):
        self.tracks = defaultdict(lambda: {
            'positions': [],
            'poses': [],
            'velocities': [],
            'last_seen': 0,
            'state': 'normal'  # normal, falling, fighting, fallen
        })
        self.incidents = []
        self.lock = threading.Lock()

    def update(self, track_id, bbox, pose_keypoints, frame_num):
        with self.lock:
            track = self.tracks[track_id]

            # Calculate center position
            cx = (bbox.left + bbox.width / 2)
            cy = (bbox.top + bbox.height / 2)

            track['positions'].append((cx, cy, frame_num))
            track['poses'].append(pose_keypoints)
            track['last_seen'] = frame_num

            # Keep only last 30 frames of history
            if len(track['positions']) > 30:
                track['positions'] = track['positions'][-30:]
                track['poses'] = track['poses'][-30:]

            # Detect actions
            self._detect_fall(track_id, track, bbox)
            self._detect_fight(track_id, track)

    def _detect_fall(self, track_id, track, bbox):
        """Detect falls based on pose and motion analysis"""
        if len(track['positions']) < 5:
            return

        # Check for rapid vertical movement (falling)
        positions = track['positions'][-10:]
        if len(positions) >= 2:
            start_y = positions[0][1]
            end_y = positions[-1][1]
            vertical_movement = end_y - start_y

            # Check aspect ratio change (person becoming horizontal)
            aspect_ratio = bbox.width / max(bbox.height, 1)

            # Fall detection criteria:
            # 1. Significant downward movement
            # 2. Aspect ratio > 1 (wider than tall = lying down)
            if vertical_movement > 50 and aspect_ratio > 1.2:
                if track['state'] != 'fallen':
                    track['state'] = 'falling'
                    self._record_incident(track_id, 'FALL',
                        f"Person {track_id} detected falling at position ({bbox.left:.0f}, {bbox.top:.0f})")
                    track['state'] = 'fallen'

    def _detect_fight(self, track_id, track):
        """Detect fights based on rapid erratic movement and proximity"""
        if len(track['positions']) < 10:
            return

        # Calculate velocity variance (erratic movement = potential fight)
        positions = track['positions'][-10:]
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append((dx**2 + dy**2)**0.5)

        if velocities:
            avg_velocity = sum(velocities) / len(velocities)
            velocity_variance = sum((v - avg_velocity)**2 for v in velocities) / len(velocities)

            # High velocity variance + high average = fight-like motion
            if velocity_variance > 500 and avg_velocity > 20:
                # Check for nearby persons (fights involve multiple people)
                nearby_count = self._count_nearby_persons(track_id)
                if nearby_count > 0 and track['state'] != 'fighting':
                    track['state'] = 'fighting'
                    self._record_incident(track_id, 'FIGHT',
                        f"Potential fight detected involving person {track_id} and {nearby_count} others")

    def _count_nearby_persons(self, track_id):
        """Count persons within proximity of the given track"""
        if track_id not in self.tracks:
            return 0

        my_pos = self.tracks[track_id]['positions'][-1] if self.tracks[track_id]['positions'] else None
        if not my_pos:
            return 0

        count = 0
        for other_id, other_track in self.tracks.items():
            if other_id == track_id or not other_track['positions']:
                continue
            other_pos = other_track['positions'][-1]
            distance = ((my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2)**0.5
            if distance < 150:  # Within 150 pixels
                count += 1
        return count

    def _record_incident(self, track_id, incident_type, description):
        """Record an incident"""
        incident = {
            'timestamp': datetime.now().isoformat(),
            'type': incident_type,
            'track_id': track_id,
            'description': description
        }
        self.incidents.append(incident)

        # Log to file
        log_file = LOGS_DIR / f"incidents_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(incident) + '\n')

        print(f"\n{'='*60}")
        print(f"[INCIDENT] {incident_type} DETECTED!")
        print(f"Time: {incident['timestamp']}")
        print(f"Description: {description}")
        print(f"{'='*60}\n")

        # Trigger recording (would integrate with recording pipeline)
        self._trigger_recording(incident)

    def _trigger_recording(self, incident):
        """Trigger video recording of incident"""
        # Recording is handled by the GStreamer pipeline
        # This would signal to save the buffer
        pass

    def get_recent_incidents(self, limit=10):
        with self.lock:
            return self.incidents[-limit:]

# Global tracker instance
tracker = PersonTracker()


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Probe function to process each frame"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Only process person detections (class_id 0 for PeopleNet)
            if obj_meta.class_id == 0:
                track_id = obj_meta.object_id
                rect_params = obj_meta.rect_params

                # Get pose keypoints if available from secondary inference
                pose_keypoints = []
                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                        # Extract pose data if available
                    except StopIteration:
                        break
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                # Create bbox object for tracker
                class BBox:
                    def __init__(self, left, top, width, height):
                        self.left = left
                        self.top = top
                        self.width = width
                        self.height = height

                bbox = BBox(
                    rect_params.left,
                    rect_params.top,
                    rect_params.width,
                    rect_params.height
                )

                # Update tracker
                tracker.update(track_id, bbox, pose_keypoints, frame_number)

                # Add incident overlay if detected
                track = tracker.tracks.get(track_id)
                if track and track['state'] in ['falling', 'fighting', 'fallen']:
                    display_text = f"[{track['state'].upper()}]"
                    obj_meta.text_params.display_text = display_text
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # Red border
                    obj_meta.rect_params.border_width = 4

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def create_pipeline(sources):
    """Create the DeepStream pipeline"""
    Gst.init(None)

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return None

    # Create elements
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    tee = Gst.ElementFactory.make("tee", "tee")

    # Display sink
    queue_display = Gst.ElementFactory.make("queue", "queue-display")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    # Recording sink
    queue_record = Gst.ElementFactory.make("queue", "queue-record")
    nvvidconv_record = Gst.ElementFactory.make("nvvideoconvert", "converter-record")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    parser = Gst.ElementFactory.make("h264parse", "parser")
    muxer = Gst.ElementFactory.make("mp4mux", "muxer")
    filesink = Gst.ElementFactory.make("filesink", "filesink")

    if not all([streammux, pgie, tracker, sgie, nvvidconv, nvosd, tee,
                queue_display, sink, queue_record, nvvidconv_record,
                encoder, parser, muxer, filesink]):
        sys.stderr.write("Unable to create elements\n")
        return None

    # Configure streammux
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', len(sources))
    streammux.set_property('batched-push-timeout', 40000)

    # Configure primary inference (PeopleNet)
    pgie.set_property('config-file-path', '/app/config/config_infer_primary_peoplenet.txt')

    # Configure tracker
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', '/app/config/config_tracker_NvDCF_perf.txt')
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)

    # Configure secondary inference (body pose)
    sgie.set_property('config-file-path', '/app/config/config_infer_secondary_bodypose.txt')

    # Configure display sink
    sink.set_property('sync', False)

    # Configure recording
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filesink.set_property('location', f'/app/recordings/recording_{timestamp}.mp4')
    encoder.set_property('bitrate', 4000000)

    # Add elements to pipeline
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(tee)
    pipeline.add(queue_display)
    pipeline.add(sink)
    pipeline.add(queue_record)
    pipeline.add(nvvidconv_record)
    pipeline.add(encoder)
    pipeline.add(parser)
    pipeline.add(muxer)
    pipeline.add(filesink)

    # Add sources
    for i, source_uri in enumerate(sources):
        print(f"Creating source {i}: {source_uri}")
        if source_uri.startswith('rtsp://'):
            source = Gst.ElementFactory.make("rtspsrc", f"source-{i}")
            source.set_property('location', source_uri)
            source.set_property('latency', 100)
            decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{i}")
        else:
            source = Gst.ElementFactory.make("filesrc", f"source-{i}")
            source.set_property('location', source_uri)
            decoder = Gst.ElementFactory.make("decodebin", f"decoder-{i}")

        pipeline.add(source)
        pipeline.add(decoder)

        # Connect source to streammux
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        if source_uri.startswith('rtsp://'):
            source.connect('pad-added', on_rtspsrc_pad_added, decoder)
        else:
            source.link(decoder)
        decoder.connect('pad-added', on_decoder_pad_added, sinkpad)

    # Link pipeline
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)

    # Link display branch
    tee.link(queue_display)
    queue_display.link(sink)

    # Link recording branch
    tee.link(queue_record)
    queue_record.link(nvvidconv_record)
    nvvidconv_record.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    # Add probe for analytics
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    return pipeline


def on_rtspsrc_pad_added(src, pad, decoder):
    """Handle RTSP source pad addition"""
    caps = pad.get_current_caps()
    struct = caps.get_structure(0)
    if struct.get_name().startswith('application/x-rtp'):
        sinkpad = decoder.get_static_pad('sink')
        if not sinkpad.is_linked():
            pad.link(sinkpad)


def on_decoder_pad_added(decoder, pad, sinkpad):
    """Handle decoder pad addition"""
    caps = pad.get_current_caps()
    if caps:
        struct = caps.get_structure(0)
        if struct.get_name().startswith('video/'):
            if not sinkpad.is_linked():
                pad.link(sinkpad)


def bus_call(bus, message, loop):
    """Handle pipeline messages"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    return True


def main():
    # Ensure directories exist
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Get sources from arguments or use default
    if len(sys.argv) > 1:
        sources = sys.argv[1:]
    else:
        # Default to sample video
        sources = ['/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4']

    print("="*60)
    print("SF Security Camera - Fall & Fight Detection")
    print("="*60)
    print(f"Sources: {sources}")
    print(f"Recordings: {RECORDINGS_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print("="*60)

    pipeline = create_pipeline(sources)
    if not pipeline:
        sys.exit(1)

    # Create event loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start pipeline
    print("\nStarting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")

    pipeline.set_state(Gst.State.NULL)

    # Print incident summary
    incidents = tracker.get_recent_incidents(limit=100)
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Total incidents detected: {len(incidents)}")
    for inc in incidents:
        print(f"  [{inc['type']}] {inc['timestamp']} - {inc['description']}")
    print("="*60)


if __name__ == '__main__':
    main()
