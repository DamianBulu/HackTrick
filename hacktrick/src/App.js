import './App.css';
import Sidebar from "./Sidebar/Sidebar";
import { useEffect, useMemo, useRef, useState } from "react";
import { FaPause, FaPlay } from "react-icons/fa6";
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

function App() {
    const FPS = 30;
    const [matchData, setMatchData] = useState(null);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
    const [inputValue, setInputValue] = useState("");
    const [chartKey, setChartKey] = useState(0);
    const [messages, setMessages] = useState([
        { sender: "Coach", message: "What would you suggest?" },
        { sender: "AI", message: "Momentum curve updated. The ball trajectory in this phase shows deep defensive pressure." },
    ]);
    const [play, setPlay] = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [viewMode, setViewMode] = useState('pitch');
    const [currentVideoPath, setCurrentVideoPath] = useState('/Data/vid3_fixed.mp4');

    const [events] = useState([
        {
            type: 'Press',
            title: 'High press trigger',
            desc: 'Deac + Boateng press Hindrich.',
            color: '#4cc9f0',
            videoPath: '/Data/vid3_fixed.mp4',
            jsonPath: '/Data/vid1_results.json'
        },
        {
            type: 'Press',
            title: 'Midfield shape collapses',
            desc: 'Ofosu 3m out of position.',
            color: '#4cc9f0',
            videoPath: '/Data/vid3_fixed.mp4',
            jsonPath: '/Data/vid1_results.json'
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

    const videoRef = useRef(null);
    const timerRef = useRef(null);

    const formatTime = (frameIndex) => {
        const totalSeconds = Math.floor(frameIndex / FPS);
        const mins = Math.floor(totalSeconds / 60);
        const secs = totalSeconds % 60;
        return `${mins}' ${secs.toString().padStart(2, '0')}"`;
    };

    const timeToFrame = (timeStr) => {
        if (!timeStr) return 0;
        const regex = /(\d+)'\s*(\d+)"/;
        const match = timeStr.match(regex);
        if (match) {
            const mins = parseInt(match[1]);
            const secs = parseInt(match[2]);
            return (mins * 60 + secs) * FPS;
        }
        return 0;
    };

    const totalFrames = matchData ? matchData.predictions.frames.length : 0;

    // FORCES RECALCULATION: Momentum data strictly follows the current matchData
    const momentumData = useMemo(() => {
        if (!matchData) return [];
        const rawPoints = matchData.predictions.frames.map(frame => {
            const ballBoxes = frame.boxes.filter(b => b.cls_id === 0);
            if (ballBoxes.length > 0) {
                const bestBall = ballBoxes.reduce((p, c) => (p.conf > c.conf ? p : c));
                // Ball x-position normalized to -50 (Defense) and +50 (Offense)
                return ((bestBall.x1 + bestBall.x2) / 3840) * 100 - 50;
            }
            return null;
        });

        const smoothed = [];
        const windowSize = 30;
        for (let i = 0; i < rawPoints.length; i++) {
            const window = rawPoints.slice(Math.max(0, i - windowSize), i + 1).filter(v => v !== null);
            smoothed.push(window.length > 0 ? window.reduce((a, b) => a + b, 0) / window.length : (smoothed[i-1] || 0));
        }
        return smoothed;
    }, [matchData]);

    const players = useMemo(() => {
        if (!matchData || !matchData.predictions.frames[currentFrameIndex]) return [];
        const boxes = matchData.predictions.frames[currentFrameIndex].boxes;

        // Pick only the single best ball detection
        const ballBoxes = boxes.filter(b => b.cls_id === 0);
        const bestBall = ballBoxes.length > 0
            ? [ballBoxes.reduce((best, b) => b.conf > best.conf ? b : best)]
            : [];

        // Players/referees must have a track_id
        const playerBoxes = boxes.filter(b => (b.cls_id === 6 || b.cls_id === 7) && b.track_id !== null);

        return [...bestBall, ...playerBoxes].map(box => {
            const isBall = box.cls_id === 0;
            return {
                id: isBall ? 'ball' : box.track_id.toString(),
                x: (((box.x1 + box.x2) / 2 / 1920) * 100) - 50,
                y: (((box.y1 + box.y2) / 2 / 1080) * 100) - 50,
                label: isBall ? '' : box.track_id.toString(),
                color: isBall ? '#ffffff' : (box.cls_id === 7 ? '#d4af37' : '#e63946'),
                isBall
            };
        });
    }, [matchData, currentFrameIndex]);

    const chartData = {
        labels: momentumData.map((_, i) => formatTime(i)),
        datasets: [{
            data: momentumData,
            borderColor: '#00c9a7',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            backgroundColor: 'rgba(0, 201, 167, 0.1)'
        }],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500 }, // Added visual transition when the graph bends
        scales: {
            x: { grid: { display: false }, ticks: { color: '#888', font: { size: 10 }, autoSkip: true, maxTicksLimit: 8 } },
            y: { min: -50, max: 50, display: false }
        },
        plugins: { legend: { display: false } },
        onClick: (event, elements, chart) => {
            const canvasPosition = getRelativePosition(event.native || event, chart);
            const dataX = chart.scales.x.getValueForPixel(canvasPosition.x);
            const targetFrame = Math.floor(dataX);
            if (targetFrame >= 0 && targetFrame < totalFrames) setCurrentFrameIndex(targetFrame);
        }
    };

    const handleEventClick = async (event) => {
        setPlay(false);
        clearInterval(timerRef.current);
        if (videoRef.current) videoRef.current.pause();

        try {
            const response = await fetch(event.jsonPath);
            const newData = await response.json();

            setCurrentVideoPath(event.videoPath);
            setMatchData(newData);
            setCurrentFrameIndex(0);           // always start at frame 0
            setChartKey(k => k + 1);           // force chart remount every time

        } catch (e) {
            console.error("JSON Load Error", e);
        }
    };

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
                    setCurrentFrameIndex(p => p >= totalFrames - 1 ? (clearInterval(timerRef.current), setPlay(false), p) : p + 1);
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
                                        <div className="center-line" /><div className="center-circle" />
                                        <div className="penalty-area-left" /><div className="penalty-area-right" />
                                    </div>
                                    {players.map(p => (
                                        <div key={p.id} className="player-marker" style={{ left: `${50 + p.x}%`, top: `${50 + p.y}%`, backgroundColor: p.color, width: p.isBall ? '12px' : '26px', height: p.isBall ? '12px' : '26px', zIndex: p.isBall ? 10 : 1, border: p.isBall ? '1px solid black' : '2px solid white' }}>{p.label}</div>
                                    ))}
                                </div>
                            ) : (
                                <div className="video-container">
                                    <video ref={videoRef} key={currentVideoPath} src={currentVideoPath} muted playsInline className="match-video" onTimeUpdate={() => play && setCurrentFrameIndex(Math.floor(videoRef.current.currentTime * FPS))} onEnded={() => setPlay(false)} />
                                </div>
                            )}
                        </div>

                        <div className="timeline-section">
                            <div className="timeline-header">
                                <strong>BALL MOMENTUM (Global)</strong>
                                <span>CLOCK: {formatTime(currentFrameIndex)}</span>
                            </div>
                            <div className="graph-container" style={{ height: '80px', position: 'relative' }}>
                                {/* THE KEY: Adding currentVideoPath to the key forces React to completely re-render the chart component, ensuring the new momentum "shape" is drawn */}
                                <Line key={chartKey} data={chartData} options={chartOptions} />
                                <div className="chart-cursor" style={{ left: `${(currentFrameIndex / (totalFrames - 1)) * 100}%` }} />
                            </div>
                            <div className="controls-row">
                                <button className="play-btn" onClick={toggleFrames}>{play ? <FaPause /> : <FaPlay />} {play ? "Pause" : "Play"}</button>
                                <select className="speed-dropdown" value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}>
                                    {[0.5, 0.75, 1, 1.25, 2].map(s => <option key={s} value={s}>{s}x</option>)}
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
                                    <input value={inputValue} type="text" placeholder="Ask AI..." onChange={(e) => setInputValue(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && sendMessage()} />
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