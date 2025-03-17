Get started:
``` bash
conda create -n ui python=3.12
pip install -r requirements.txt
flask run
```

go to localhost:5000 in your browser.

Hints:
- ./docs stores all documents you want to display. You can change this in doc-lib.js: 14
- in app.py: 23, you can add your own ai reply.
- Now only MD file can be displayed. you can modify this in app.py: 74. Note that you must handle how to display the file of the file type you specify.
- canvasOperation.js: 353 for neo4j configuration.