import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { readFileSync, readdirSync, statSync, watch } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ port: 8765 });

const RECORDINGS_DIR = join(__dirname, 'public', 'recordings');
const LOGS_DIR = join(__dirname, 'logs');

// Serve static files
app.use(express.static(join(__dirname)));
app.use('/recordings', express.static(RECORDINGS_DIR));

// API: Get recent incidents
app.get('/api/incidents', (req, res) => {
    try {
        const today = new Date().toISOString().split('T')[0].replace(/-/g, '');
        const logFile = join(LOGS_DIR, `incidents_${today}.jsonl`);

        try {
            const content = readFileSync(logFile, 'utf8');
            const incidents = content.trim().split('\n')
                .filter(line => line)
                .map(line => JSON.parse(line))
                .reverse()
                .slice(0, 50);
            res.json(incidents);
        } catch (e) {
            res.json([]);
        }
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// API: Get recordings
app.get('/api/recordings', (req, res) => {
    try {
        const files = readdirSync(RECORDINGS_DIR)
            .filter(f => f.endsWith('.mp4'))
            .map(f => {
                const stats = statSync(join(RECORDINGS_DIR, f));
                return {
                    name: f,
                    size: formatSize(stats.size),
                    timestamp: stats.mtime.toISOString()
                };
            })
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, 20);
        res.json(files);
    } catch (e) {
        res.json([]);
    }
});

// API: Get stats
app.get('/api/stats', (req, res) => {
    try {
        const today = new Date().toISOString().split('T')[0].replace(/-/g, '');
        const logFile = join(LOGS_DIR, `incidents_${today}.jsonl`);

        let falls = 0;
        let fights = 0;

        try {
            const content = readFileSync(logFile, 'utf8');
            content.trim().split('\n').forEach(line => {
                if (!line) return;
                const incident = JSON.parse(line);
                if (incident.type === 'FALL') falls++;
                if (incident.type === 'FIGHT') fights++;
            });
        } catch (e) {
            // No log file yet
        }

        res.json({ falls, fights, people: 0, cameras: 4 });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// WebSocket for real-time updates
const clients = new Set();

wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('Client connected');

    ws.on('close', () => {
        clients.delete(ws);
        console.log('Client disconnected');
    });
});

function broadcast(message) {
    const data = JSON.stringify(message);
    clients.forEach(client => {
        if (client.readyState === 1) {
            client.send(data);
        }
    });
}

// Watch for new incidents
try {
    watch(LOGS_DIR, (eventType, filename) => {
        if (filename && filename.endsWith('.jsonl')) {
            // Read last line and broadcast
            try {
                const content = readFileSync(join(LOGS_DIR, filename), 'utf8');
                const lines = content.trim().split('\n');
                const lastLine = lines[lines.length - 1];
                if (lastLine) {
                    const incident = JSON.parse(lastLine);
                    broadcast({
                        type: 'incident',
                        incident,
                        stats: {
                            falls: lines.filter(l => l.includes('FALL')).length,
                            fights: lines.filter(l => l.includes('FIGHT')).length
                        }
                    });
                }
            } catch (e) {
                // Ignore read errors
            }
        }
    });
} catch (e) {
    console.log('Could not watch logs directory:', e.message);
}

// Watch for new recordings
try {
    watch(RECORDINGS_DIR, (eventType, filename) => {
        if (filename && filename.endsWith('.mp4')) {
            try {
                const stats = statSync(join(RECORDINGS_DIR, filename));
                broadcast({
                    type: 'recording',
                    recording: {
                        name: filename,
                        size: formatSize(stats.size),
                        timestamp: stats.mtime.toISOString()
                    }
                });
            } catch (e) {
                // File might still be writing
            }
        }
    });
} catch (e) {
    console.log('Could not watch recordings directory:', e.message);
}

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Nimverse Dashboard running at http://localhost:${PORT}`);
    console.log(`WebSocket server running on port 8765`);
});
