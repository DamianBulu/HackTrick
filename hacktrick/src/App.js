import './App.css';
import Sidebar from "./Sidebar/Sidebar";
import {useEffect, useRef, useState} from "react";
import * as messages from "react-bootstrap/ElementChildren";

function App() {
    const [players] = useState([
        { id: 'BOR', x: -46, y: 0, label: 'BOR', color: '#d4af37' },
        { id: 'CAM', x: -40, y: 42, label: 'CAM', color: '#d4af37' },
        { id: 'PAU', x: -35, y: 25, label: 'PAU', color: '#d4af37' },
        { id: 'OFO', x: -25, y: 35, label: 'OFO', color: '#d4af37' },
        { id: '9', x: -10, y: 5, label: '9', color: '#e63946' },
        { id: '8', x: 8, y: 5, label: '8', color: '#e63946' },
        { id: '1', x: 45, y: 0, label: '1', color: '#e63946' },
        { id: 'DEA', x: 15, y: -40, label: 'DEA', color: '#d4af37' },
    ]);

    const [events] = useState([
        { type: 'Press', title: 'High press trigger', time: "12'", desc: 'Deac + Boateng press Hindrich. Ball forced long, Cociuc wins header at 28m.', color: '#4cc9f0' },
        { type: 'Press', title: 'Midfield shape collapses', time: "34'", desc: 'Ofosu 3m out of position. Kone + Yuri combine through gap.', color: '#4cc9f0' },
        { type: 'Counter', title: 'Counter — CFR goal (1-1)', time: "38'", desc: 'Paun steps out, 8m CB gap. Omrani through in 4.2s. Worst transition.', color: '#ff4d4d' },
        { type: 'Counter', title: 'Counter — CFR goal (1-1)', time: "38'", desc: 'Paun steps out, 8m CB gap. Omrani through in 4.2s. Worst transition.', color: '#ff4d4d' }
    ]);

    const [inputValue, setInputValue] = useState("");

    const [messages, setMessages] = useState([
        {
            sender: "Coach",
            message: "What would you suggest?"
        },
        {
            sender: "AI",
            message: "Based on the match data: the patterns you're asking about are visible in the momentum curve. Scrub to the relevant minute to see the exact tactical frame."
        },
        {
            sender: "Coach",
            message: "Ok, what else?"
        },
        {
            sender: "AI",
            message: "Nu stiu boss"
        },
    ])

    const chatEndRef = useRef(null);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const sendMessage = () => {
        if(inputValue === "") {
            return;
        }
        const message = {
            sender: "Coach",
            message: inputValue,
        }
        setMessages([...messages, message]);
        setInputValue("");

        setTimeout(() => {
            setMessages(prev => [...prev, {
                sender: "AI",
                message: "Analizăm faza imediat..."
            }]);
        }, 1000);
    }

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    return (
        <div className="main-body">
            <div className="main-container">
                <Sidebar />

                <div className="dashboard-content">
                    {/* LEFT: PITCH & MOMENTUM */}
                    <div className="main-visual-column">
                        <div className="pitch-section">
                            <div className="pitch-container">
                                <div className="field-lines">
                                    <div className="center-line"></div>
                                    <div className="center-circle"></div>
                                    <div className="penalty-area-left"></div>
                                    <div className="penalty-area-right"></div>
                                    <div className="goal-gate-left"></div>
                                    <div className="goal-gate-right"></div>
                                </div>

                                {players.map(p => (
                                    <div
                                        key={p.id}
                                        className="player-marker"
                                        style={{
                                            left: `${50 + p.x}%`,
                                            top: `${50 + p.y}%`,
                                            backgroundColor: p.color
                                        }}
                                    >
                                        {p.label}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="timeline-section">
                            <div className="timeline-header">
                                <strong>MOMENTUM TIMELINE — CONTROL SCORE</strong>
                                <span>79' · control: +0.09 · 4-4-2 Defensive block</span>
                            </div>
                            <div className="graph-container">
                                <svg width="100%" height="60">
                                    <path d="M0 40 Q 100 10, 200 45 T 400 30" fill="none" stroke="#00c9a7" strokeWidth="2" />
                                </svg>
                            </div>
                            <div className="timeline-controls">
                                <button className="play-btn">▶ Play</button>
                                <input type="range" className="scrubber" />
                                <span className="time-display">66'</span>
                            </div>
                        </div>
                    </div>

                    {/* RIGHT: EVENTS & CHAT */}
                    <div className="info-column">
                        <div className="events-container">
                            <h4 className="section-title">KEY EVENTS</h4>
                            <div className="events-list">
                                {events.map((ev, i) => (
                                    <div key={i} className="event-item">
                                        <div className="event-meta">
                                            <span className="tag" style={{backgroundColor: ev.color}}>{ev.type}</span>
                                            <span className="event-name">{ev.title}</span>
                                            <span className="timestamp">{ev.time}</span>
                                        </div>
                                        <p className="event-description">{ev.desc}</p>
                                        <div className="event-dots">
                                            {[1, 2, 3].map(d => <span key={d} className="dot" style={{backgroundColor: ev.color}}></span>)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* AI ANALYST Section */}
                        <h4 className="section-title">AI ANALYST</h4>
                        <div className="chat-section">
                            <div className="chat-messages">
                                {messages.map((msg, i) => (
                                    <div
                                        key={i}
                                        className={`message-wrapper ${msg.sender === "Coach" ? "align-right" : "align-left"}`}
                                    >
                                        <div className={`message ${msg.sender}`}>
                                            <div className="sender-name">{msg.sender === "AI" ? "AI analyst" : "Coach"}</div>
                                            <div className="message-text">{msg.message}</div>
                                        </div>
                                    </div>
                                ))}
                                <div ref={chatEndRef} />
                            </div>

                            <div className="chat-input-wrapper">
                                <div className="input-pill">
                                    <input value={inputValue} type="text" placeholder="Ask about this match..." onChange={(e) => setInputValue(e.target.value)}/>
                                    <button className="send-btn" onClick={() => {sendMessage()}}>Send</button>
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