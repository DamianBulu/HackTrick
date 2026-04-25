import './Sidebar.css'
import {useState} from "react";
import {FaPlus} from "react-icons/fa";

function Sidebar() {
    const [active, setActive] = useState(1);
    const mock = [
        {
            "id": 1,
            "match" : "CFR",
            "competitie" : "Liga 1",
            "data" : "14 Apr 2026"
        },
        {
            "id": 2,
            "match" : "Universitatea Craiova",
            "competitie" : "Cupa Romaniei",
            "data" : "22 Apr 2026"
        }
    ]

    return (

        <div className={"sidebar"}>
            <div className="sidebar-content">
                {
                    mock.map((elem, index) => (
                        <div key={elem.id} className={active === elem.id ? "card active_card" : "card"} onClick={() => setActive(elem.id)}>
                            <div>{elem.match}</div>
                            <small className={"card_subtitle"}>{elem.competitie} <br/> {elem.data} </small>
                        </div>
                    ))
                }
                <div className={"add-button"}>
                    <FaPlus style={{marginRight: '5px'}}/>
                    <div>
                        Add match
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Sidebar;