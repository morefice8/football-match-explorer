# Match Explorer ⚽

Match Explorer is a comprehensive and interactive football match analysis dashboard, built with Python, Dash, and Plotly. It allows analysts and enthusiasts to dive deep into event data from a match, from formations and passing networks to transition phases and individual player statistics.

---

## 🚀 Key Features

### Navigation and Selection
- **Dynamic Home Page**: Filter matches by league, season, team, or round.
- **Interactive Match Cards**: Quickly view key information for each match and access the analysis with a single click.
- **League Analysis**: A dedicated section to compare teams within a league on advanced statistical metrics.

### In-Depth Match Analysis
The application offers a sidebar with several analysis sections for each match:

- **📋 Match Overview**: Explore and filter the complete table of all match events, with an option to download the data as a CSV file.
- **♟️ Formation**:
    - **Formation Timeline**: Visualize formation and player position changes throughout the match, synchronized with key events (goals, substitutions).
    - **Average Positions**: Charts showing the average on-field positions of players.
- **↔️ Passes**:
    - **Pass Network**: Interactive passing networks to visualize connections between players.
    - **Progressive Passes**: Maps highlighting progressive passes for each team.
    - **Final Third Entries**: Detailed analysis of passes entering the final third, including "Zone 14" and the half-spaces.
    - **Pass Locations**: Heatmaps and density maps to visualize the most used areas of the pitch for passing.
    - **Crosses**: A dedicated analysis of crosses, with filters for origin/destination zones, cross type, and a Sankey diagram to visualize flows.
- **📈 Buildup**: An interactive sequence explorer to analyze offensive buildup phases, with filters for outcome, flank, and buildup type.
- **🛡️ Defensive Transition**: Analysis of the defensive transition phase, including charts of the defensive block, compactness (convex hull), PPDA (Passes Per Defensive Action), and a carousel to explore conceded counter-attacks.
- **⚡ Offensive Transition**: Analysis of the offensive transition, with a sequence explorer showing how a team attacks immediately after regaining possession.
- **⛳ Set Piece**: A section dedicated to analyzing set pieces (corners and free kicks), with filters for type, zone, trajectory, and outcome.
- **🧑‍🚀 Player Analysis**: Detailed individual analysis divided into:
    - **Passing**: Stats on the top passers and individual pass maps.
    - **Shooting**: Charts on involvement in shooting sequences and maps of passes received by the most dangerous players.
    - **Defending**: Stats on the top defenders and interactive maps showing the location of all their defensive actions.

---

## 🛠️ Tech Stack
- **Backend & Web Framework**: [Dash](https://dash.plotly.com/) (on top of [Flask](https://flask.palletsprojects.com/))
- **Data Visualization**: [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/)
- **UI Components**: [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) (SLATE Theme)
- **Icons**: [Font Awesome](https://fontawesome.com/)

---

## 📁 Data Structure
For the application to work correctly, the data must be organized according to the following folder structure:
.
├── app.py # Main Dash application file
├── requirements.txt # List of Python dependencies
├── assets/ # Static files (CSS, images, logos)
│ └── logos/
├── data/ # Raw and processed data (see Installation and Setup section)
└── src/ # Project source code
├── init.py
├── config.py # Configuration variables (colors, mappings, paths)
├── data_processing/ # Modules for data cleaning and preparation
├── metrics/ # Modules for calculating metrics
├── utils/ # Utility functions (e.g., loading mappings)
└── visualization/ # Modules for creating plots


---

## ⚙️ Installation and Setup

Follow these steps to set up and run the project locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/morefice8/football-match-explorer.git
    cd football-match-explorer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unzip the Match Data (Crucial Step):**
    The raw match data files are too large to be stored directly in the repository. They have been compressed into zip archives. You need to extract them for the app to work.
    
    - Navigate to the following directory in your local project folder:
      ```
      data/matches/English Premier League/2024-2025/partidos/
      ```
    - Inside this folder, you will find four `.zip` files: `data_part_1.zip`, `data_part_2.zip`, `data_part_3.zip`, and `data_part_4.zip`.
    - **Extract all of them** directly into this same folder. Your file manager (like Windows Explorer or macOS Finder) should have an "Extract All" or "Unzip" option.
    - After extraction, the `partidos` folder should contain all the individual `.json` match files.

5.  **Run the application:**
    ```bash
    python app.py
    ```

6.  Open your browser and navigate to `http://127.0.0.1:8050/`.

> **Note:** The `data/` folder is listed in the `.gitignore` file. This is intentional to prevent large data files from being tracked by Git. The setup process requires you to manually unzip the provided data archives after cloning the repository.

---

## 🔮 Future Improvements
- [ ] Improve the report generation feature to create more complex PDF or HTML files.
- [ ] Make the league/season selection in the "League Analysis" page dynamic.
- [ ] Add unit and integration tests to ensure code stability.
- [ ] Optimize data loading for matches with a large volume of events.

---

## 📜 License
This project is distributed under the MIT License. See the `LICENSE` file for more details.
