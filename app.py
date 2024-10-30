from flask import Flask, render_template, redirect, url_for, session, request, send_from_directory
from scipy.stats import chi2_contingency
import io
import base64
from matplotlib.figure import Figure
import numpy as np
import os

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

results_data = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/start_trial', methods=['POST'])
def start_trial():
    # Get user inputs from the form
    participants = int(request.form.get('participants', 10))
    plane_height = float(request.form.get('plane_height', 3000))
    deploy_height = float(request.form.get('deploy_height', 1000))
    weather = request.form.get('weather', 'no_wind')

    # Ensure deploy_height is less than plane_height
    deploy_height = min(deploy_height, plane_height)

    session['participants'] = participants
    session['plane_height'] = plane_height
    session['deploy_height'] = deploy_height
    session['weather'] = weather

    return redirect(url_for('simulate'))

@app.route('/simulate')
def simulate():
    participants = session.get('participants', 10)
    plane_height = session.get('plane_height', 3000)
    deploy_height = session.get('deploy_height', 1000)
    weather = session.get('weather', 'no_wind')  # Retrieve weather from session

    # Reset results_data for a new simulation
    results_data.clear()

    parachute_masses = []
    parachute_velocities = []
    placebo_masses = []
    placebo_velocities = []

    for _ in range(participants):
        # Randomly assign parachute or placebo
        group = np.random.choice(['parachute', 'placebo'])
        result = physics_simulation(group, plane_height, deploy_height)
        # Store the result
        results_data.append({
            'group': group,
            'mass': result['mass'],
            'impact_velocity': result['impact_velocity'],
            'outcome': result['outcome']
        })
        # Collect data for averages
        if group == 'parachute':
            parachute_masses.append(result['mass'])
            parachute_velocities.append(result['impact_velocity'])
        else:
            placebo_masses.append(result['mass'])
            placebo_velocities.append(result['impact_velocity'])

    # Calculate aggregate results
    parachute_results, placebo_results = calculate_aggregate_results()
    parachute_total = sum(parachute_results.values())
    placebo_total = sum(placebo_results.values())

    # Perform chi-squared test
    p_value = perform_chi_squared_test(parachute_results, placebo_results)

    # Create bar chart
    chart_data = create_bar_chart(parachute_results, placebo_results)

    # Calculate averages
    avg_parachute_mass = np.mean(parachute_masses) if parachute_masses else 0
    avg_placebo_mass = np.mean(placebo_masses) if placebo_masses else 0
    avg_parachute_velocity = np.mean(parachute_velocities) if parachute_velocities else 0
    avg_placebo_velocity = np.mean(placebo_velocities) if placebo_velocities else 0

    return render_template('results.html',
                           participants=participants,
                           plane_height=plane_height,
                           deploy_height=deploy_height,
                           weather=weather,
                           parachute_results=parachute_results,
                           placebo_results=placebo_results,
                           parachute_total=parachute_total,
                           placebo_total=placebo_total,
                           p_value=p_value,
                           chart_data=chart_data,
                           avg_parachute_mass=avg_parachute_mass,
                           avg_placebo_mass=avg_placebo_mass,
                           avg_parachute_velocity=avg_parachute_velocity,
                           avg_placebo_velocity=avg_placebo_velocity)

def physics_simulation(group, plane_height, deploy_height):
    import random
    # Physics parameters
    g = 9.81  # Acceleration due to gravity (m/s^2)

    # Randomize mass between 50 kg and 120 kg
    mass = random.uniform(50, 120)
    mass = round(mass, 2)  # Round to two decimal places

    # Get weather condition from session
    weather = session.get('weather', 'no_wind')

    # Adjust parameters based on weather
    if weather == 'no_wind':
        wind_factor = 1.0
    elif weather == 'fair_wind':
        wind_factor = 1.1  # Slight increase in impact
    elif weather == 'strong_wind':
        wind_factor = 1.2  # Significant increase in impact
    else:
        wind_factor = 1.0

    # Ensure deploy_height is within bounds
    deploy_height = max(0, min(deploy_height, plane_height))

    # Free fall until deploy_height
    fall_distance = plane_height - deploy_height
    free_fall_time = np.sqrt(2 * fall_distance / g)
    free_fall_velocity = g * free_fall_time  # Velocity at deploy_height

    if group == 'parachute':
        # Descent rate under parachute
        impact_velocity = np.random.normal(5, 1) * wind_factor  # Adjust for wind

        # Ensure impact_velocity is positive
        impact_velocity = max(0, impact_velocity)

        # Total time of fall
        parachute_descent_time = (deploy_height) / impact_velocity
        total_time = free_fall_time + parachute_descent_time

        # Adjust outcome probabilities based on wind
        # Probabilities adjusted based on real-world data
        if weather == 'no_wind':
            probs = [0.9983, 0.0017, 0.00002, 0.000005]
        elif weather == 'fair_wind':
            probs = [0.9973, 0.0026, 0.00009, 0.00001]
        elif weather == 'strong_wind':
            probs = [0.995, 0.004, 0.0009, 0.0001]
        probs = normalize_probabilities(probs)

        # Outcome probabilities
        outcome = np.random.choice(
            ['No Injury', 'Minor Injury', 'Serious Injury', 'Fatality'],
            p=probs
        )
    else:
        # No parachute, continue free fall
        terminal_velocity = 53  # Approximate terminal velocity in m/s

        # Time to reach terminal velocity
        t_terminal = terminal_velocity / g

        # Distance covered until reaching terminal velocity
        d_terminal = 0.5 * g * t_terminal**2

        if d_terminal < plane_height:
            # Time falling at terminal velocity
            d_remaining = plane_height - d_terminal
            t_remaining = d_remaining / terminal_velocity
            total_time = t_terminal + t_remaining
        else:
            # Never reaches terminal velocity
            total_time = np.sqrt(2 * plane_height / g)

        impact_velocity = terminal_velocity * wind_factor  # Adjust for wind

        # Outcome probabilities
        outcome = np.random.choice(
            ['Fatality', 'Serious Injury'],
            p=[0.9999, 0.0001]
        )

    result = {
        'mass': mass,
        'impact_velocity': round(impact_velocity, 2),
        'time_of_fall': round(total_time, 2),
        'outcome': outcome
    }

    return result

def normalize_probabilities(probs):
    total = sum(probs)
    return [p / total for p in probs]

def calculate_aggregate_results():
    from collections import Counter

    parachute_outcomes = [r['outcome'] for r in results_data if r['group'] == 'parachute']
    placebo_outcomes = [r['outcome'] for r in results_data if r['group'] == 'placebo']

    parachute_counts = Counter(parachute_outcomes)
    placebo_counts = Counter(placebo_outcomes)

    # Possible outcomes
    outcomes = ['No Injury', 'Minor Injury', 'Serious Injury', 'Fatality']
    parachute_results = {k: parachute_counts.get(k, 0) for k in outcomes}
    placebo_results = {k: placebo_counts.get(k, 0) for k in outcomes}

    return parachute_results, placebo_results

def perform_chi_squared_test(parachute_results, placebo_results):
    # For the chi-squared test, we'll consider 'Survived' vs 'Fatality'
    parachute_survived = (
        parachute_results['No Injury'] +
        parachute_results['Minor Injury'] +
        parachute_results['Serious Injury']
    )
    parachute_fatalities = parachute_results['Fatality']
    placebo_survived = (
        placebo_results['No Injury'] +
        placebo_results['Minor Injury'] +
        placebo_results['Serious Injury']
    )
    placebo_fatalities = placebo_results['Fatality']

    contingency_table = np.array([
        [parachute_survived, parachute_fatalities],
        [placebo_survived, placebo_fatalities]
    ], dtype=float)  # Convert to float type

    # Add a small value to avoid division by zero
    if contingency_table.min() == 0:
        contingency_table += 0.5

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return p_value

def create_bar_chart(parachute_results, placebo_results):
    outcomes = ['No Injury', 'Minor Injury', 'Serious Injury', 'Fatality']
    parachute_counts = [parachute_results[o] for o in outcomes]
    placebo_counts = [placebo_results[o] for o in outcomes]

    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    x = np.arange(len(outcomes))
    width = 0.35

    ax.bar(x - width/2, parachute_counts, width, label='Parachute')
    ax.bar(x + width/2, placebo_counts, width, label='Placebo')
    ax.set_ylabel('Number of Participants')
    ax.set_title('Outcomes by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()

    # Convert plot to PNG image
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    chart_data = base64.b64encode(img.getvalue()).decode()
    return chart_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
