from flask import Flask
from flask import render_template

## Test
sample_name = 'MESHANE'
# End of test


# Flask application
app = Flask(__name__)

# Landing page
@app.route('/')
def home():
    return render_template('index.html', name=sample_name)

if __name__ == "__main__":
    app.run(debug=True)

