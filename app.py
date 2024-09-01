from flask import Flask
from flask import render_template
from api.api import past_month_titles, past_month_descriptions

## Test
sample_name = 'MESHANE'
# End of test


# Flask application
app = Flask(__name__)

# Landing page
@app.route('/')
def home():
    return render_template('index.html', name=sample_name)

@app.route('/latestnews')
def latestnews():
    return render_template('latestnews.html', titles=past_month_titles, descriptions=past_month_descriptions, numArticles=len(past_month_titles))

@app.route('/landingpage')
def landingpage():
    return render_template('landingpage.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == "__main__":
    app.run(debug=True)


