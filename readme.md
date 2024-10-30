# Parachute Effectiveness Trial Simulator

A web-based simulator that demonstrates the effectiveness of parachutes in preventing death and major trauma related to gravitational challenge. This project is inspired by the satirical BMJ paper ["Parachute use to prevent death and major trauma related to gravitational challenge: systematic review of randomised controlled trials"](https://pubmed.ncbi.nlm.nih.gov/14684649/).

## 🎯 About

This is a concept demonstration that simulates a double-blind trial testing parachute effectiveness. It uses simplified physics models and probability distributions for educational purposes. The simulation allows users to:

- Set the number of participants
- Choose plane height
- Set parachute deployment height
- Select weather conditions

The results include:
- Outcome distribution visualization
- Statistical analysis (p-value)
- Impact velocity calculations
- Mass distribution analysis

## ⚠️ Disclaimer

This is an educational concept demonstration and not hard science. The simulation uses simplified physics models and probability distributions for illustrative purposes only.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Magnussmari/Parachute_trial.git
cd Parachute_trial
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:10000
```

## 🛠️ Built With

- Flask - Web framework
- NumPy - Numerical computations
- Matplotlib - Data visualization
- SciPy - Statistical analysis

## 📚 Citation

If you use this simulator in your work, please cite both this repository and the original paper that inspired it:

```bibtex
@article{smith2003parachute,
  title={Parachute use to prevent death and major trauma related to gravitational challenge: systematic review of randomised controlled trials},
  author={Smith, Gordon CS and Pell, Jill P},
  journal={BMJ},
  volume={327},
  number={7429},
  pages={1459--1461},
  year={2003},
  publisher={British Medical Journal Publishing Group}
}
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Smith GC, Pell JP's BMJ paper
- Built by Magnús Smári Smárason
- Website: [www.smarason.is](https://www.smarason.is)
