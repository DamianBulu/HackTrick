import './App.css';
import Sidebar from "./Sidebar/Sidebar";
import { useEffect, useMemo, useRef, useState } from "react";
import {FaBackwardStep, FaForwardStep, FaPause, FaPlay} from "react-icons/fa6";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler
} from 'chart.js';
import { getRelativePosition } from 'chart.js/helpers';
import { Line } from "react-chartjs-2";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler
);

// ── Pitch coordinate system ───────────────────────────────────────────────────
// Matches the Python Tracker2D: pitch is 105m × 68m, origin at top-left.
// Pixel bounds come from the actual tracking data (barca3_results_complete.json).
// These convert raw pixel centres → pitch metres, then → CSS % from top-left.
const PITCH_W_M   = 105.0;
const PITCH_H_M   = 68.0;
const FIELD_X_MIN = 155;
const FIELD_X_MAX = 1907;
const FIELD_Y_MIN = 238;
const FIELD_Y_MAX = 1025;

// px  →  metres  (0..105, 0..68) — same space the Python tracker works in
const toMetresX = (px) => ((px - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)) * PITCH_W_M;
const toMetresY = (py) => ((py - FIELD_Y_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)) * PITCH_H_M;

// metres  →  CSS % from top-left corner of the pitch div (0..100)
const toCssX = (mx) => (mx / PITCH_W_M) * 100;
const toCssY = (my) => (my / PITCH_H_M) * 100;

// ── Hungarian algorithm (Munkres) ─────────────────────────────────────────────
// Minimal O(n³) implementation — no external library needed.
// Returns [rowIndices, colIndices] matched pairs, just like scipy linear_sum_assignment.
function hungarianAssign(costMatrix) {
    const n = costMatrix.length;
    const m = costMatrix[0]?.length ?? 0;
    if (n === 0 || m === 0) return [[], []];

    // Pad to square
    const size = Math.max(n, m);
    const C = Array.from({ length: size }, (_, i) =>
        Array.from({ length: size }, (_, j) =>
            i < n && j < m ? costMatrix[i][j] : 1e9
        )
    );

    const u = new Array(size + 1).fill(0);
    const v = new Array(size + 1).fill(0);
    const p = new Array(size + 1).fill(0); // col → row assignment
    const way = new Array(size + 1).fill(0);

    for (let i = 1; i <= size; i++) {
        p[0] = i;
        let j0 = 0;
        const minVal = new Array(size + 1).fill(Infinity);
        const used = new Array(size + 1).fill(false);

        do {
            used[j0] = true;
            const i0 = p[j0];
            let delta = Infinity;
            let j1 = -1;
            for (let j = 1; j <= size; j++) {
                if (!used[j]) {
                    const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minVal[j]) { minVal[j] = cur; way[j] = j0; }
                    if (minVal[j] < delta) { delta = minVal[j]; j1 = j; }
                }
            }
            for (let j = 0; j <= size; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minVal[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] !== 0);

        do {
            const j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    const rowInd = [], colInd = [];
    for (let j = 1; j <= size; j++) {
        if (p[j] !== 0 && p[j] - 1 < n && j - 1 < m) {
            rowInd.push(p[j] - 1);
            colInd.push(j - 1);
        }
    }
    return [rowInd, colInd];
}

// ── Re-ID Tracker — direct port of Python ReIDTracker ────────────────────────
// Assigns stable IDs across frames using Hungarian matching per class.
// Tracks survive MAX_AGE frames of non-detection before being pruned.
class ReIDTracker {
    constructor() {
        this.MAX_DIST_M = 5.0;  // metres — matches Python
        this.MAX_AGE    = 10;   // frames — matches Python
        this._tracks    = {};   // { cls_id: { stableId: { pos:[x,y], age:int } } }
        this._nextId    = 1;
    }

    // detections: [ { cls_id, mx, my } ]
    // returns:    stable ID per detection (same order)
    update(detections) {
        // Group by class
        const byCls = {};
        detections.forEach((d, i) => {
            (byCls[d.cls_id] ??= []).push({ i, mx: d.mx, my: d.my });
        });

        // Age all tracks by 1 before matching
        for (const cls of Object.keys(this._tracks)) {
            for (const sid of Object.keys(this._tracks[cls])) {
                this._tracks[cls][sid].age++;
            }
        }

        const stableIds = new Array(detections.length).fill(null);

        for (const [cls, items] of Object.entries(byCls)) {
            if (!this._tracks[cls]) this._tracks[cls] = {};
            const tracks    = this._tracks[cls];
            const trackSids = Object.keys(tracks);
            const trackPos  = trackSids.map(sid => tracks[sid].pos);

            if (trackSids.length === 0) {
                // No existing tracks — spawn all as new
                for (const { i, mx, my } of items) {
                    const sid = this._nextId++;
                    tracks[sid] = { pos: [mx, my], age: 0 };
                    stableIds[i] = sid;
                }
                continue;
            }

            // Build cost matrix (n_det × n_tracks)
            const costMatrix = items.map(({ mx, my }) =>
                trackPos.map(([tx, ty]) =>
                    Math.hypot(mx - tx, my - ty)
                )
            );

            const [rowInd, colInd] = hungarianAssign(costMatrix);

            const matchedDets = new Set();
            for (let k = 0; k < rowInd.length; k++) {
                const ri = rowInd[k], ci = colInd[k];
                if (costMatrix[ri][ci] <= this.MAX_DIST_M) {
                    const sid = trackSids[ci];
                    const { mx, my } = items[ri];
                    tracks[sid].pos = [mx, my];
                    tracks[sid].age = 0;
                    stableIds[items[ri].i] = parseInt(sid);
                    matchedDets.add(ri);
                }
            }

            // Unmatched detections → new tracks
            items.forEach(({ i, mx, my }, ri) => {
                if (!matchedDets.has(ri)) {
                    const sid = this._nextId++;
                    tracks[sid] = { pos: [mx, my], age: 0 };
                    stableIds[i] = sid;
                }
            });
        }

        // Prune stale tracks
        for (const cls of Object.keys(this._tracks)) {
            for (const sid of Object.keys(this._tracks[cls])) {
                if (this._tracks[cls][sid].age > this.MAX_AGE) {
                    delete this._tracks[cls][sid];
                }
            }
        }

        return stableIds;
    }

    reset() {
        this._tracks = {};
        this._nextId = 1;
    }
}

// ── App ───────────────────────────────────────────────────────────────────────
function App() {
    const FPS = 30;
    const [matchData,         setMatchData]         = useState(null);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
    const [inputValue,        setInputValue]        = useState("");
    const [chartKey,          setChartKey]          = useState(0);
    const [messages, setMessages] = useState([
        { sender: "Coach", message: "What would you suggest?" },
        { sender: "AI",    message: "Momentum curve updated. The ball trajectory in this phase shows deep defensive pressure." },
    ]);
    const [play,          setPlay]          = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [viewMode,      setViewMode]      = useState('pitch');
    const [currentVideoPath, setCurrentVideoPath] = useState('/Data/vid3_fixed.mp4');
    const [activeEventIndex, setActiveEventIndex] = useState(0);

    const [events] = useState([
        {
            type: 'Press',
            title: 'High press trigger',
            desc: 'Deac + Boateng press Hindrich.',
            color: '#4cc9f0',
            videoPath: '/Data/barca3_fixed.mp4',
            jsonPath: '/Data/barca3_results_complete.json'
        },
        {
            type: 'Press',
            title: 'Midfield shape collapses',
            desc: 'Ofosu 3m out of position.',
            color: '#4cc9f0',
            videoPath: '/Data/barca5_fixed.mp4',
            jsonPath: '/Data/barca5_results_complete.json'
        },
        {
            type: 'Counter',
            title: 'Counter — CFR goal (1-1)',
            desc: 'Paun steps out, 8m CB gap.',
            color: '#ff4d4d',
            videoPath: '/Data/vid3_fixed.mp4',
            jsonPath: '/Data/vid3_results.json'
        },
    ]);

    const videoRef   = useRef(null);
    const timerRef   = useRef(null);

    const PITCH_W_M = 105.0;
    const PITCH_H_M = 68.0;
    const toCssX = (mx) => (mx / PITCH_W_M) * 100;
    const toCssY = (my) => (my / PITCH_H_M) * 100;

    const FULL_MATCH_MINUTES = 90;
    const TOTAL_MATCH_MINUTES = 90;
    const TOTAL_MATCH_FRAMES = TOTAL_MATCH_MINUTES * 60 * FPS; // 162,000 frames

    // Re-ID tracker instance — persists for the lifetime of the component
    const reidRef    = useRef(new ReIDTracker());

    // ── Reset tracker when match data changes ─────────────────────────────
    useEffect(() => {
        reidRef.current.reset();
    }, [matchData]);

    // ── Time helpers ──────────────────────────────────────────────────────
    const formatTime = (frameIndex) => {
        // Determine which minute offset to use based on which file is loaded
        const offsetSeconds = matchData?.predictions?.frames?.[0]?.frame_id === 0 && matchData.predictions.frames.length > 500
            ? 15 * 60  // Offset for barca3
            : 27 * 60; // Offset for barca5

        const totalSeconds = Math.floor(frameIndex / FPS) + offsetSeconds;
        const mins = Math.floor(totalSeconds / 60);
        const secs = totalSeconds % 60;
        return `${mins}' ${secs.toString().padStart(2, '0')}"`;
    };

    const totalFrames = matchData ? matchData.predictions.frames.length : 0;

    // ── Momentum graph data ───────────────────────────────────────────────
    // Ball x-position in metres, smoothed with a 30-frame rolling average.
    // Python equivalent: the ball's pitch_x projected by homography.
    // We don't have keypoints here, so we use pixel→metres conversion instead.
    const momentumData = useMemo(() => {
        if (!matchData) return [];
        const rawPoints = matchData.predictions.frames.map(frame => {
            const ballBoxes = frame.boxes.filter(b => b.cls_id === 0 && b.pitch_x != null);
            if (ballBoxes.length === 0) return null;
            const best = ballBoxes.reduce((p, c) => c.conf > p.conf ? c : p);
            return best.pitch_x;
        });

        const smoothed = [];
        const W = 30;
        for (let i = 0; i < rawPoints.length; i++) {
            const window = rawPoints
                .slice(Math.max(0, i - W), i + 1)
                .filter(v => v !== null);
            smoothed.push(
                window.length > 0
                    ? window.reduce((a, b) => a + b, 0) / window.length
                    : (smoothed[i - 1] ?? PITCH_W_M / 2)
            );
        }
        return smoothed;
    }, [matchData]);

    // ── Remove these — no longer needed for player/ball positioning ───────────
// const FIELD_X_MIN = 155; ...toMetresX, toMetresY are no longer used

// ── Keep these — they map real metres → CSS % ────────────────────────────

    // ── Player positions — Hungarian Re-ID tracker ────────────────────────
    //
    // Direct port of Python's ReIDTracker.update() logic:
    //   1. Convert pixel centres → pitch metres (same coordinate space).
    //   2. Feed into the ReIDTracker which runs Hungarian matching per class.
    //   3. Tracker keeps stable IDs across ID-resets and short gaps (MAX_AGE=10).
    //
    // The tracker is STATEFUL (lives in reidRef) — it remembers last-seen
    // positions so IDs survive brief occlusions, exactly like the Python version.
    //
    // Note: we process frames sequentially as currentFrameIndex advances.
    // Scrubbing backwards resets the tracker via the useEffect above.
    const players = useMemo(() => {
        if (!matchData || !matchData.predictions.frames[currentFrameIndex]) return [];

        const CONF_THRESHOLD = 0.35; // matches Python _CONF_THRESHOLD
        const frame = matchData.predictions.frames[currentFrameIndex];

        // ── Ball: single best detection, no ID tracking needed ────────────
        const ballBoxes = frame.boxes.filter(b => b.cls_id === 0);
        const bestBall  = ballBoxes.length > 0
            ? ballBoxes.reduce((best, b) => b.conf > best.conf ? b : best)
            : null;

        // ── Players / GK / Ref: confidence gate, then Re-ID ──────────────
        const validBoxes = frame.boxes.filter(
            b => [1, 3, 6, 7].includes(b.cls_id) && b.conf >= CONF_THRESHOLD
        );

        // Convert to pitch metres — this is what Python's _project() returns
        const detections = validBoxes
            .filter(b => b.pitch_x != null) // only use boxes with real homography coords
            .map(b => ({
                cls_id: b.cls_id,
                mx: b.pitch_x,   // ← real metres, no conversion needed
                my: b.pitch_y,
            }));

        // Hungarian matching → stable IDs (same algorithm as Python ReIDTracker)
        const stableIds = reidRef.current.update(detections);

        // ── Build CSS-ready player markers ────────────────────────────────
        const colorByCls = { 1: '#4CAF50', 3: '#888888', 6: '#e63946', 7: '#ffffff' };

        const playerMarkers = detections.map((det, i) => ({
            id:     stableIds[i]?.toString() ?? `u${i}`,
            // toCssX/Y maps metres → 0..100% from top-left, matching the pitch div layout
            x:      toCssX(det.mx),
            y:      toCssY(det.my),
            label:  stableIds[i]?.toString() ?? '?',
            color:  colorByCls[det.cls_id] ?? '#aaaaaa',
            isBall: false,
        }));

        const ballMarker = bestBall ? [{
            id: 'ball',
            x: toCssX(bestBall.pitch_x),
            y: toCssY(bestBall.pitch_y),
            label:  '',
            color:  '#ffffff',
            isBall: true,
        }] : [];

        return [...ballMarker, ...playerMarkers];
    }, [matchData, currentFrameIndex]);

    // ── Chart ─────────────────────────────────────────────────────────────
    const chartData = {
        labels: momentumData.map((_, i) => formatTime(i)),
        datasets: [{
            data: momentumData,
            borderColor: '#00c9a7',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            backgroundColor: 'rgba(0, 201, 167, 0.1)',
        }],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'linear', // Ensure it treats frames as a continuous number line
                min: 0,
                max: TOTAL_MATCH_FRAMES,
                grid: { display: false },
                ticks: {
                    color: '#888',
                    font: { size: 10 },
                    maxTicksLimit: 10, // Shows labels roughly every 10 minutes
                    callback: function(value) {
                        // value is the frame index.
                        // Convert frame index -> total seconds -> minutes
                        const minutes = Math.floor(value / (FPS * 60));
                        return minutes + "'";
                    }
                },
            },
            y: {
                min: 0,
                max: 105,
                display: false
            },
        },
        plugins: { legend: { display: false } },
        onClick: (event, elements, chart) => {
            const pos = getRelativePosition(event.native || event, chart);
            const targetFrame = Math.floor(chart.scales.x.getValueForPixel(pos.x));

            // Only jump if the frame exists in our current JSON data
            if (targetFrame >= 0 && targetFrame < totalFrames) {
                setCurrentFrameIndex(targetFrame);
            }
        },
    };

    // ── Event click ───────────────────────────────────────────────────────
    const handleEventClick = async (event) => {
        setPlay(false);
        if (timerRef.current) clearInterval(timerRef.current);
        if (videoRef.current) videoRef.current.pause();

        try {
            const response = await fetch(event.jsonPath);
            const newData  = await response.json();

            // Find which event we just loaded to set the correct 15' or 27' offset
            const idx = events.findIndex(e => e.jsonPath === event.jsonPath);
            setActiveEventIndex(idx);

            setCurrentVideoPath(event.videoPath);
            setMatchData(newData);
            setCurrentFrameIndex(0);
            setChartKey(k => k + 1);
        } catch (e) {
            console.error("Error loading JSON:", e);
        }
    };

    // ── Playback ──────────────────────────────────────────────────────────
    const toggleFrames = () => {
        if (play) {
            if (videoRef.current) videoRef.current.pause();
            clearInterval(timerRef.current);
            setPlay(false);
        } else {
            setPlay(true);
            if (viewMode === 'video' && videoRef.current) {
                videoRef.current.playbackRate = playbackSpeed;
                videoRef.current.play();
            } else {
                timerRef.current = setInterval(() => {
                    setCurrentFrameIndex(p =>
                        p >= totalFrames - 1
                            ? (clearInterval(timerRef.current), setPlay(false), p)
                            : p + 1
                    );
                }, (1000 / FPS) / playbackSpeed);
            }
        }
    };

    const sendMessage = () => {
        if (!inputValue.trim()) return;
        setMessages([...messages, { sender: "Coach", message: inputValue }]);
        setInputValue("");
    };

    useEffect(() => {
        if (viewMode === 'video' && videoRef.current && !play) {
            videoRef.current.currentTime = currentFrameIndex / FPS;
        }
    }, [currentFrameIndex, viewMode, play]);

    useEffect(() => { if (events.length > 0) handleEventClick(events[0]); }, []);

    const getGlobalOffset = () => {
        // If activeEventIndex is 0 (Barca 3), offset is 15 mins.
        // If 1 (Barca 5), offset is 27 mins.
        if (activeEventIndex === 0) return 15 * 60 * FPS;
        if (activeEventIndex === 1) return 67 * 60 * FPS;
        return 0;
    };

    const getYPos = (min) => {
        const offsetFrames = getGlobalOffset();
        const targetGlobalFrame = min * 60 * FPS;

        // Calculate which local frame in the current clip corresponds to this global minute
        const localFrame = targetGlobalFrame - offsetFrames;

        // Get the ball value (0-105m). If the frame isn't in this clip, default to center (52.5)
        const val = (localFrame >= 0 && localFrame < momentumData.length)
            ? momentumData[localFrame]
            : 52.5;

        // Convert meters to percentage (0-100).
        // We do 100 - x because in CSS 'top: 0' is the top of the box.
        return 100 - (val / 105 * 100);
    };

    if (!matchData) return <div className="loading-screen">Synthesizing Pitch Data...</div>;

    return (
        <div className="main-body">
            <div className="main-container">
                <Sidebar />
                <div className="dashboard-content">
                    <div className="main-visual-column">
                        <div className="pitch-section">
                            <div className="view-toggle">
                                <button className={viewMode === 'pitch' ? 'active' : ''} onClick={() => setViewMode('pitch')}>Tactical</button>
                                <button className={viewMode === 'video' ? 'active' : ''} onClick={() => setViewMode('video')}>Video</button>
                            </div>
                            {viewMode === 'pitch' ? (
                                <div className="pitch-container">
                                    <div className="field-lines">
                                        <div className="center-line" />
                                        <div className="center-circle" />
                                        <div className="penalty-area-left" />
                                        <div className="penalty-area-right" />
                                    </div>
                                    {players.map(p => (
                                        <div
                                            key={p.id}
                                            className="player-marker"
                                            style={{
                                                // x/y are already 0–100% from top-left
                                                left:            `${p.x}%`,
                                                top:             `${p.y}%`,
                                                backgroundColor: p.color,
                                                width:           p.isBall ? '12px' : '26px',
                                                height:          p.isBall ? '12px' : '26px',
                                                zIndex:          p.isBall ? 10 : 1,
                                                border:          p.isBall ? '1px solid black' : '2px solid white',
                                                transform:       'translate(-50%, -50%)',
                                            }}
                                        >
                                            {p.label}
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="video-container">
                                    <video
                                        ref={videoRef}
                                        key={currentVideoPath}
                                        src={currentVideoPath}
                                        muted
                                        playsInline
                                        className="match-video"
                                        onTimeUpdate={() => play && setCurrentFrameIndex(Math.floor(videoRef.current.currentTime * FPS))}
                                        onEnded={() => setPlay(false)}
                                    />
                                </div>
                            )}
                        </div>

                        <div className="timeline-section">
                            <div className="timeline-header">
                                <strong>BALL MOMENTUM (Global)</strong>
                                <span>CLOCK: {formatTime(currentFrameIndex)}</span>
                            </div>
                            <div className="graph-container" style={{ height: '80px', position: 'relative' }}>
                                <Line key={chartKey} data={chartData} options={chartOptions} />

                                {/* 15' Red Triangle */}
                                <div
                                    onClick={() => handleEventClick(events[0])} // This now updates activeEventIndex
                                    style={{
                                        position: 'absolute',
                                        left: `${(15 / 90) * 100}%`,
                                        top: `${getYPos(15)}%`,
                                        cursor: 'pointer',
                                        width: '0', height: '0',
                                        borderLeft: '7px solid transparent',
                                        borderRight: '7px solid transparent',
                                        borderBottom: '12px solid #ff4d4d',
                                        transform: 'translate(-50%, -50%)',
                                        zIndex: 30
                                    }}
                                />

                                {/* 27' Blue Square */}
                                <div
                                    onClick={() => handleEventClick(events[1])}
                                    style={{
                                        position: 'absolute',
                                        left: `${(67 / 90) * 100}%`,
                                        top: `${getYPos(77)}%`,
                                        cursor: 'pointer',
                                        width: '10px', height: '10px',
                                        backgroundColor: '#4cc9f0',
                                        transform: 'translate(-50%, -50%)',
                                        zIndex: 30,
                                        border: '1px solid white'
                                    }}
                                />

                                {/* Current Playback Cursor */}
                                <div
                                    className="chart-cursor"
                                    style={{
                                        // Uses the new activeEventIndex-based offset
                                        left: `${((getGlobalOffset() + currentFrameIndex) / TOTAL_MATCH_FRAMES) * 100}%`,
                                        pointerEvents: 'none',
                                        position: 'absolute',
                                        top: 0,
                                        height: '100%',
                                        width: '2px',
                                        backgroundColor: 'white',
                                        zIndex: 35
                                    }}
                                />
                            </div>
                            <div className="controls-row">
                                {/* Skip to Start of Phase */}
                                <button
                                    className="control-btn"
                                    onClick={() => setCurrentFrameIndex(0)}
                                    title="Skip to start of phase"
                                >
                                    <FaBackwardStep />
                                </button>

                                <button className="play-btn" onClick={toggleFrames}>
                                    {play ? <FaPause /> : <FaPlay />} {play ? "Pause" : "Play"}
                                </button>

                                {/* Skip to End of Phase */}
                                <button
                                    className="control-btn"
                                    onClick={() => setCurrentFrameIndex(totalFrames - 1)}
                                    title="Skip to end of phase"
                                >
                                    <FaForwardStep />
                                </button>

                                <select
                                    className="speed-dropdown"
                                    value={playbackSpeed}
                                    onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                                >
                                    {[0.5, 0.75, 1, 1.25, 2].map(s => (
                                        <option key={s} value={s}>{s}x</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>

                    <div className="info-column">
                        <div className="events-container">
                            <h4 className="section-title">KEY EVENTS</h4>
                            <div className="events-list">
                                {events.map((ev, i) => (
                                    <div key={i} className="event-item clickable-event" onClick={() => handleEventClick(ev)}>
                                        <div className="event-meta">
                                            <span className="tag" style={{ backgroundColor: ev.color }}>{ev.type}</span>
                                            <span className="event-name">{ev.title}</span>
                                            <span className="timestamp">{ev.time}</span>
                                        </div>
                                        <p className="event-description">{ev.desc}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <h4 className="section-title">AI ANALYST</h4>
                        <div className="chat-section">
                            <div className="chat-messages">
                                {messages.map((msg, i) => (
                                    <div key={i} className={`message-wrapper ${msg.sender === "Coach" ? "align-right" : "align-left"}`}>
                                        <div className={`message ${msg.sender}`}>
                                            <div className="sender-name">{msg.sender === "AI" ? "AI analyst" : "Coach"}</div>
                                            <div className="message-text">{msg.message}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="chat-input-wrapper">
                                <div className="input-pill">
                                    <input
                                        value={inputValue}
                                        type="text"
                                        placeholder="Ask AI..."
                                        onChange={(e) => setInputValue(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;