import './Sidebar.css'
import { useState } from "react";
import { FaPlus } from "react-icons/fa";

function Sidebar() {
    const [active, setActive] = useState(1);
    const [isHovered, setIsHovered] = useState(false);

    const mock = [
        {
            "id": 1,
            "match": "Barcelona",
            "shortcut": "FCB",
            "competitie": "Champions League",
            "data": "14 Apr 2026"
        }
    ];

    return (
        <div
            className={`sidebar ${isHovered ? 'expanded' : 'collapsed'}`}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <div className="sidebar-content">
                <div className="cards-wrapper">
                    {mock.map((elem) => (
                        <div
                            key={elem.id}
                            className={active === elem.id ? "card active_card" : "card"}
                            onClick={() => setActive(elem.id)}
                        >
                            <div className="card-title">
                                {/* Use acronym when collapsed, full name when expanded */}
                                {isHovered ? elem.match : elem.shortcut}
                            </div>

                            {/* Only show extra info when hovered/expanded */}
                            {isHovered && (
                                <small className="card_subtitle">
                                    {elem.competitie} <br/> {elem.data}
                                </small>
                            )}
                        </div>
                    ))}
                </div>

                <div className="add-button">
                    <FaPlus className="add-icon" />
                    {isHovered && <span>Add match</span>}
                </div>
            </div>
        </div>
    );
}

export default Sidebar;