===================================================
  Seed Node Identification in Erdős-Rényi Graphs
===================================================

PROJECT STRUCTURE
-----------------
seed_nodes_er_graph/
├── demo.ipynb              ← Main notebook (run this)
├── run_experiments.py      ← Importable functions 
├── algorithms/
│   ├── bp.py               ← Belief Propagation (EM + BP)
│   ├── baseline.py         ← NetFill baseline
│   └── ilp.py              ← ILP / MAP estimation
├── simulation/
│   └── simulator.py        ← SI model simulation on ER graphs
└── utils/
    ├── metrics.py          ← Precision / Recall / F1
    └── plotting.py         ← Bar chart & runtime chart helpers


DEPENDENCIES
------------
    pip install pyscipopt networkx torch scipy numpy pandas matplotlib

Note: pyscipopt requires a SCIP installation.
      On Colab, `pip install pyscipopt` usually installs everything automatically.


==================================================
  HOW TO RUN LOCALLY (Jupyter)
==================================================

1. Install dependencies:
       pip install pyscipopt networkx torch scipy numpy pandas matplotlib

2. Launch Jupyter from the project root folder:
       jupyter notebook demo.ipynb
          OR
       jupyter lab demo.ipynb

3. In the notebook:
   - SKIP Cell 0 (Colab setup)
   - Edit parameters in Cell "Experiment Parameters" (N, NUM_SEEDS, etc.)
   - Run All Cells:  Kernel > Restart & Run All

4. Results are saved automatically to:
       results_N<N>_K<NUM_SEEDS>_p<P_NOISE>.csv
5. Yo


==================================================
  HOW TO RUN ON GOOGLE COLAB
==================================================

Step 1 — Go to https://colab.research.google.com and open a new notebook.

Step 2 — Upload the project (choose ONE option):

  Option A — Google Drive (recommended):
    a. Upload the folder to your Google Drive.
    b. In Colab, run:
         from google.colab import drive
         drive.mount('/content/drive')
         import os
         os.chdir('/content/drive/MyDrive/seed_nodes_er_graph')

  Option B — Zip upload:
    a. Zip the folder: seed_nodes_er_graph.zip
    b. In Colab, run:
         from google.colab import files
         files.upload()            # select seed_nodes_er_graph.zip
         !unzip seed_nodes_er_graph.zip -d /content/
         import os
         os.chdir('/content/seed_nodes_er_graph')

Step 3 — Open demo.ipynb from the Colab file browser (left panel).

Step 4 — Run Cell 0 ("Colab Setup"):
    - Uncomment the lines matching your upload option (A or B).
    - The cell also runs:  !pip install -q pyscipopt ...

Step 5 — Edit parameters in the "Experiment Parameters" cell if needed.

Step 6 — Run All Cells:  Runtime > Run all  (or Ctrl+F9)

Step 7 — Download results (optional):
    The last cell in the notebook contains a download snippet:
         from google.colab import files
         files.download(CSV_FILE)
    Uncomment and run it to save the CSV to your computer.



