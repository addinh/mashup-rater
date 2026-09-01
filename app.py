import os
import uuid
import json
from flask import Flask, Response, request, jsonify, send_file, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from scipy.io import wavfile

from audioGenerator import generateInstVocalsMashupMetadata, SR
import pandas as pd

# Initialize Flask app and setup
app = Flask(__name__)
with open('/etc/config.json') as config_file:
  config = json.load(config_file)

basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'instance', 'audio_ratings.db')

app.config['SECRET_KEY'] = config.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'temp_audio'

db = SQLAlchemy(app)

class Track(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mtd = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

class User(db.Model):
    id = db.Column(db.String(64), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(64), db.ForeignKey('user.id'))
    track_id = db.Column(db.Integer, db.ForeignKey('track.id'))

    # rating values
    rating_overall = db.Column(db.Integer)
    rating_harmony = db.Column(db.Integer)
    rating_essence = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>User ID Input</title>
        <style>
            .container { max-width: 400px; margin: 100px auto; padding: 20px; text-align: center; }
            form { margin: 20px 0; }
            input { width: 200px; padding: 5px; margin: 10px 0; }
            button { padding: 8px 16px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Audio Rating System</h1>
            <p>Please enter your 8-digit Student ID. Make sure your ID number is correct, otherwise you won't get credits for completing this survey.</p>
            <form id="user-id-form">
                <input type="text" id="user-id" placeholder="Enter 8-digit Student ID" required pattern="\d{8}">
                <button type="submit">Submit ID</button>
            </form>
            <button onclick="window.location.href='/set_id/0'" type="button">Start Without ID</button>
        </div>
        <script>
            document.getElementById('user-id-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const userId = document.getElementById('user-id').value;
                if (userId.length === 8 && /^\d+$/.test(userId)) {
                    window.location.href = `/set_id/${userId}`;
                } else {
                    alert('Please enter a valid 8-digit Student ID');
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/set_id/<string:user_id>')
def set_user_id(user_id):
    resp = app.make_response(redirect(url_for('rating_page')))
    resp.set_cookie('user_id', user_id)
    return resp

@app.route('/rating_page')
def rating_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Random Mashup Rater</title>
        <style>
            .container { max-width: 700px; margin: auto; padding: 20px; }
            .rating-form { margin-top: 20px; }
            input[type="range"] { width: 100%; }
            .loading { color: #666; }
            .welcome-message { font-size: 1.2em; margin-bottom: 10px; }
            .rating-count { font-size: 1.1em; color: #666; margin-bottom: 10px; }
            .pyramid-container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 20px;
                max-width: 650px; /* Forces the third item to wrap to the next row */
                margin: 0 auto;
            }
            .audio-item {
                display: flex;
                flex-direction: column; /* Places label above audio */
                align-items: center;    /* Centers label horizontally over audio */
                gap: 8px;               /* Space between label and player */
                flex: 0 1 300px;        /* Restricts player width */
            }
            .audio-item label {
                font-family: sans-serif;
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
            .rating-slider-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
                flex: 0 1 600px;
            }
            .rating-slider-item label {
                font-family: sans-serif;
                font-size: 14px;
                color: #333;
            }
            .glow-effect {
                position: relative;
                animation: glow 2s infinite;
            }
            @keyframes glow {
                0% { box-shadow: 0 0 10px rgba(0, 255, 0, 0.5); }
                50% { box-shadow: 0 0 20px rgba(0, 255, 0, 0.5); }
                100% { box-shadow: 0 0 10px rgba(0, 255, 0, 0.5); }
            }
            .contact-message { font-size: 0.8em; color: #666; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Random Mashup Rater</h1>
            <div class="welcome-message" id="welcome-message"></div>
            <div class="rating-count" id="rating-count"></div>
            <div id="loading-status" class="loading">Loading...</div>
            <br>
            <div class="pyramid-container">
                <div class="audio-item">
                    <label for="track1">Track 1 (Instrumental)</label>
                    <audio class="audio-player" id="inst-player" controls>
                        <source src="" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="audio-item">
                    <label for="track2">Track 2 (Vocals)</label>
                    <audio class="audio-player" id="vocals-player" controls>
                        <source src="" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="audio-item">
                    <label for="track3">Mashup</label>
                    <audio class="audio-player" id="mashup-player" controls>
                        <source src="" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
            </div>
            <div class="rating-form">
                <div class="rating-slider-item">
                    <label for="rating">Rate your <b>overall enjoyment</b> of this mashup from <b>1 (Disliked)</b> to <b>7 (Liked)</b>:</label>
                    <input class="rating-slider" type="range" id="rating-overall-slider" min="1" max="7" value="4" disabled oninput="document.getElementById('rating-overall-value').textContent = this.value">
                    <label id="rating-overall-value">4</label>
                </div>
                <br>
                <div class="rating-slider-item">
                    <label for="rating">Rate whether this mashup is <b>harmonically sound</b> to you from <b>1 (Disharmonious)</b> to <b>7 (Harmonious)</b>:</label>
                    <input class="rating-slider" type="range" id="rating-harmony-slider" min="1" max="7" value="4" disabled oninput="document.getElementById('rating-harmony-value').textContent = this.value">
                    <label id="rating-harmony-value">4</label>
                </div>
                <br>
                <div class="rating-slider-item">
                    <label for="rating">Rate whether this mashup contains <b>memorable elements</b> of the original tracks from <b>1 (None of them)</b> to <b>7 (All of them)</b>:</label>
                    <input class="rating-slider" type="range" id="rating-essence-slider" min="1" max="7" value="4" disabled oninput="document.getElementById('rating-essence-value').textContent = this.value">
                    <label id="rating-essence-value">4</label>
                </div>
                <button id="submit-rating" onclick="submitRating()" disabled>Submit Rating</button>
            </div>
            <br>
            <div class="contact-message">
                If you encounter issues, please contact TA Dzung: <a href="mailto:addinh@connect.ust.hk">addinh@connect.ust.hk</a>
            </div>
        </div>
        <script>
            let currentTrackId = null;
            let tracksReady = 0;

            // Audio player event listeners
            const audioElements = document.querySelectorAll('.audio-player');
            audioElements.forEach((audioElement, index) => {
                // audioElement.addEventListener('timeupdate', function() {
                //     if (this.currentTime >= 4 && !this.getAttribute('listened-enough')) {
                //         this.setAttribute('listened-enough', 'true');
                //         this.classList.add('glow-effect');
                //         tracksReady++;
                //         checkSubmitEnable();
                //     }
                // });

                audioElement.addEventListener('pause', function() {
                    if (this.currentTime >= 4 && !this.getAttribute('listened-enough')) {
                        this.setAttribute('listened-enough', 'true');
                        this.classList.add('glow-effect');
                        tracksReady++;
                        checkSubmitEnable();
                    }
                });
            });

            const submitButton = document.getElementById('submit-rating');
            function checkSubmitEnable() {
                if (tracksReady === 3) {
                    submitButton.disabled = false;
                }
            }
            
            async function getNewTrack() {
                updateWelcomeAndRatingCount();

                
                tracksReady = 0;
                submitButton.disabled = true;
                audioElements.forEach((audioElement) => {
                    audioElement.removeAttribute('listened-enough');
                    audioElement.classList.remove('glow-effect');
                });
                
                try {
                    const loadingLabel = document.getElementById('loading-status');
                    const ratingSliders = document.querySelectorAll('.rating-slider');
                    
                    loadingLabel.textContent = 'Generating audio. Please wait up to 30 seconds...';
                    ratingSliders.forEach((ratingSlider) => {
                        ratingSlider.disabled = true;
                    });

                    const response = await fetch('/generate_track');
                    const data = await response.json();
                    
                    loadingLabel.textContent = 'Loading audio...';
                    currentTrackId = data.track_id;

                    const instUrl = `/play_track/${data.inst_filename}`;
                    const instPlayer = document.getElementById('inst-player');
                    instPlayer.innerHTML = `<source src="${instUrl}" type="audio/wav">`;
                    instPlayer.load();

                    const vocalsUrl = `/play_track/${data.vocals_filename}`;
                    const vocalsPlayer = document.getElementById('vocals-player');
                    vocalsPlayer.innerHTML = `<source src="${vocalsUrl}" type="audio/wav">`;
                    vocalsPlayer.load();

                    const mashupUrl = `/play_track/${data.mashup_filename}`;
                    const mashupPlayer = document.getElementById('mashup-player');
                    mashupPlayer.innerHTML = `<source src="${mashupUrl}" type="audio/wav">`;
                    mashupPlayer.load();
                    
                    loadingLabel.textContent = 'Audio loaded!';
                    ratingSliders.forEach((ratingSlider) => {
                        ratingSlider.disabled = false;
                        ratingSlider.value = 4;
                    });
                    document.getElementById('rating-overall-value').textContent = 4;
                    document.getElementById('rating-harmony-value').textContent = 4;
                    document.getElementById('rating-essence-value').textContent = 4;

                } catch (error) {
                    alert('Error fetching track, please refresh the page. If the problem persists, contact TA Dzung at: addinh@connect.ust.hk');
                    console.error('Error fetching track:', error);
                }
            }

            async function submitRating() {
                try {
                    if (!currentTrackId) return;
                    if (tracksReady < 3) return;

                    audioElements.forEach((audioElement) => {
                        audioElement.pause();
                    });
                    
                    const response = await fetch('/submit_rating', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            track_id: currentTrackId,
                            rating_overall: document.getElementById('rating-overall-slider').value,
                            rating_harmony: document.getElementById('rating-harmony-slider').value,
                            rating_essence: document.getElementById('rating-essence-slider').value
                        })
                    });
                    
                    if (response.ok) {
                        alert('Rating submitted successfully!');
                        getNewTrack();
                    } else {
                        alert('Error submitting rating. Please contact TA Dzung at: addinh@connect.ust.hk');
                        console.error('Error submitting rating');
                    }
                } catch (error) {
                    console.error('Error submitting rating:', error);
                }
            }

            async function updateWelcomeAndRatingCount() {
                // Display welcome message and rating count
                const welcomeMessage = document.getElementById('welcome-message');
                const ratingCount = document.getElementById('rating-count');
                
                const userId = document.cookie.split('; ').find(row => row.startsWith('user_id='));
                
                if (userId) {
                    const userIdValue = userId.split('=')[1];
                    if (userIdValue === '0') {
                        welcomeMessage.textContent = 'Welcome: Guest';
                        ratingCount.textContent = 'Tracks rated: N/A';
                    } else {
                        welcomeMessage.textContent = `Welcome: ${userIdValue}`;
                        // Fetch the rating count
                        fetch(`/get_rating_count?user_id=${userIdValue}`)
                            .then(response => response.json())
                            .then(data => {
                                ratingCount.textContent = `Tracks rated: ${data.count}`;
                            })
                            .catch(error => {
                                console.error('Error fetching rating count:', error);
                            });
                    }
                } else {
                    welcomeMessage.textContent = 'Welcome: Guest';
                    ratingCount.textContent = 'Tracks rated: N/A';
                }
            }

            // Initialize with a new track
            window.onload = getNewTrack();
        </script>
    </body>
    </html>
    """

@app.route('/generate_track')
def generate_track():
    inst_data, vocals_data, mashup_data, mtd = generateInstVocalsMashupMetadata()
    inst_filename = f"{uuid.uuid4()}.wav"
    vocals_filename = f"{uuid.uuid4()}.wav"
    mashup_filename = f"{uuid.uuid4()}.wav"
    
    # Store the audio data in the temporary folder
    temp_folder = app.config['UPLOAD_FOLDER']
    wavfile.write(os.path.join(temp_folder, inst_filename), SR, inst_data)
    wavfile.write(os.path.join(temp_folder, vocals_filename), SR, vocals_data)
    wavfile.write(os.path.join(temp_folder, mashup_filename), SR, mashup_data)
    
    # Create the track in the database
    track = Track(mtd=mtd)
    db.session.add(track)
    db.session.commit()
    
    return jsonify({
        'track_id': track.id,
        'inst_filename': inst_filename,
        'vocals_filename': vocals_filename,
        'mashup_filename': mashup_filename,
    })

@app.route('/play_track/<string:filename>')
def play_track(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return 'Audio file not found', 404
    
    return send_file(
        file_path,
        mimetype='audio/wav',
        as_attachment=False,
    )

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    data = request.get_json()
    track_id = data['track_id']
    rating_overall = data['rating_overall']
    rating_harmony = data['rating_harmony']
    rating_essence = data['rating_essence']
    
    user_id = request.cookies.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID not found'}), 400
    
    # Ensure user exists
    user = User.query.get(user_id)
    if not user:
        user = User(id=user_id)
        db.session.add(user)
        db.session.commit()

    # Delete old file
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Submit rating
    # existing_rating = Rating.query.filter_by(user_id=user_id, track_id=track_id).first()
    # if existing_rating:
    #     return jsonify({'error': 'Already rated this track'}), 400
    new_rating = Rating(user_id=user_id, track_id=track_id, rating_overall=rating_overall, rating_harmony=rating_harmony, rating_essence=rating_essence)
    db.session.add(new_rating)
    db.session.commit()
    return jsonify({'message': 'Rating submitted successfully'})

@app.route('/get_rating_count')
def get_rating_count():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID not provided'}), 400
    
    if user_id == '0':
        return jsonify({'count': 0})
    
    count = Rating.query.filter_by(user_id=user_id).count()
    return jsonify({'count': count})

# @app.route('/export-csv-pandas')
# def export_csv_pandas():
#     # Execute raw sql query using the db.engine syntax
#     query = str(Rating.query.statement)
#     df = pd.read_sql_query(query, db.engine)
    
#     # Convert DataFrame to CSV string
#     csv_data = df.to_csv(index=False)
    
#     return Response(
#         csv_data,
#         mimetype="text/csv",
#         headers={"Content-Disposition": "attachment; filename=pandas_export.csv"}
#     )

# Ensure the database exists
with app.app_context():
    db.create_all()

# Create upload directory if it doesn't exist
temp_folder = app.config['UPLOAD_FOLDER']
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)


if __name__ == '__main__':
    app.run(debug=True)