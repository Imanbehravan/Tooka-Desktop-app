# Tooka Desktop

**Tooka Desktop** is an offline-first desktop application for building, training, evaluating, and managing Machine Learning models through a professional and user-friendly interface.

The application is being developed as the desktop counterpart of the **Tooka AutoML** platform, with the goal of bringing core Machine Learning capabilities from the web-based platform into a standalone Windows application.

---

## 🚀 Project Overview

Tooka AutoML is designed to simplify the Machine Learning workflow by allowing users to work with datasets, select algorithms, train models, evaluate results, and use explainability tools without having to manually implement every part of the ML pipeline.

The original platform follows a web architecture:

```text
User
 │
 ▼
Frontend
 │
 ▼
Django Backend
 │
 ▼
Machine Learning Modules
```

Tooka Desktop is being designed with a different architecture:

```text
┌─────────────────────────────┐
│       Tooka Desktop         │
│                             │
│   PySide6 / Qt Interface    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          ML Core            │
│                             │
│ Classification              │
│ Regression                  │
│ Clustering                  │
│ Time Series                 │
│ XAI                         │
└─────────────────────────────┘
```

The main idea is to **separate the Machine Learning core from the Django backend** so that the desktop application can operate independently.

---

# 🎯 Main Goals

The main goals of Tooka Desktop are:

* Provide a professional desktop interface for Tooka AutoML
* Run Machine Learning workflows locally
* Reduce dependency on an internet connection
* Separate UI, application logic, and ML logic
* Reuse the existing Machine Learning implementations
* Provide a scalable architecture for future features
* Make the application suitable for Windows deployment
* Provide an easier workflow for users who are not experienced with Machine Learning

---

# 🖥️ Target Platform

The primary target platform is:

**Windows**

The application is being developed using technologies that allow the Python application to eventually be packaged as a standalone Windows executable.

The project is currently developed and tested in a Linux development environment, while the final application is intended for Windows users.

---

# 🧠 Machine Learning Core

One of the most important architectural decisions in this project is separating the ML functionality from Django.

The existing backend contains several Machine Learning modules, including:

```text
classification/
regression/
clustering/
timeseries/
xai/
```

The reusable Machine Learning logic is being extracted into an independent **ML Core**.

The intended architecture is:

```text
Django Backend
      │
      │
      ▼
    ML Core
      │
      │
      ▼
Tooka Desktop
```

The goal is to avoid moving web-specific Django code into the desktop application.

For example, the following backend components are **not** part of the Desktop ML Core:

```text
views.py
urls.py
forms.py
Django API endpoints
HTTP request handling
```

Instead, the Desktop application communicates directly with reusable Python Machine Learning components.

---


# 🛠️ Technology Stack

## Desktop Application

* **Python**
* **PySide6**
* **Qt**
* **Qt Designer**
* **QSS**

## Machine Learning

Depending on the implemented module, the project may use:

* NumPy
* Pandas
* Scikit-learn
* PyTorch
* Transformers

## Development

* Git
* GitHub
* VS Code
* Linux development environment
* Windows as the target deployment platform

---


# 🎨 User Interface

The Tooka Desktop interface is designed with a modern desktop application experience in mind.

The visual design takes inspiration from applications such as:

* VS Code
* Notion
* Cursor
* Modern developer tools

The interface includes:

### Sidebar

Provides access to the main sections of the application.

Examples:

```text
Dashboard
AutoML
Datasets
Models
Training
Results
Settings
```

### Header

Provides application-level information and controls.

### Dashboard

The dashboard acts as the main entry point and provides an overview of:

* Projects
* Recent activity
* Model information
* Quick actions
* Statistics

### AutoML

The AutoML section is intended to provide the main Machine Learning workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Algorithm Selection
   ↓
Training
   ↓
Evaluation
   ↓
Results
   ↓
Model
```

---

# ⚙️ Background Processing

Machine Learning operations can be computationally expensive.

Running them directly on the Qt GUI thread can freeze the application.

Therefore, Tooka Desktop uses a worker/controller architecture.

For example:

```text
GUI
 │
 ▼
Controller
 │
 ▼
Worker
 │
 ▼
ML Core
```

The worker performs the training operation in a separate thread while communicating with the UI through Qt signals.

Typical signals include:

```text
progress
status
finished
error
```

This allows the UI to remain responsive while a model is being trained.

---

# 🔐 Offline Architecture

One of the main objectives of Tooka Desktop is to provide an **offline-capable Machine Learning environment**.

The application should be able to perform core ML operations locally without requiring continuous communication with the Tooka web platform.

This provides several benefits:

* Local dataset processing
* Local model training
* Faster interaction
* Reduced network dependency
* Better control over user data
* Possibility of working without an internet connection

Online services may still be introduced later for optional functionality such as synchronization, updates, licensing, or cloud deployment.

---

# 📊 Supported Machine Learning Tasks

The ML Core is planned around the following major areas:

### Classification

Examples:

* Binary classification
* Multiclass classification
* Dataset preprocessing
* Model training
* Model evaluation

### Regression

Examples:

* Linear regression
* Non-linear regression
* Model evaluation
* Prediction

### Clustering

Examples:

* Unsupervised learning
* Cluster analysis
* Visualization
* Cluster evaluation

### Time Series

Planned functionality includes:

* Time series preprocessing
* Forecasting
* Model training
* Evaluation

### Explainable AI

The XAI module is intended to provide tools for understanding model predictions and behavior.

---

# 🧪 Development Status

The project is currently under active development.

### Completed / In Progress

* [x] Initial PySide6 application
* [x] Main Window
* [x] Sidebar
* [x] Header
* [x] Dashboard
* [x] Navigation architecture
* [x] Router
* [x] Theme system
* [x] Initial AutoML page
* [x] Worker / Controller architecture
* [x] ML Core extraction
* [x] Classification integration
* [ ] Regression integration
* [ ] Clustering integration
* [ ] Time Series integration
* [ ] XAI integration
* [ ] Dataset management
* [ ] Model management
* [ ] Training interface
* [ ] Results visualization
* [ ] Windows packaging
* [ ] Installer
* [ ] Licensing and application security

---

# 🗺️ Development Roadmap

The project is being developed in multiple phases.

## Phase 1 — Desktop Foundation

```text
PySide6
Main Window
Navigation
Basic UI
```

## Phase 2 — Professional UI

```text
Sidebar
Header
Dashboard
Themes
Icons
Animations
```

## Phase 3 — Application Architecture

```text
Router
Page Manager
Controllers
Workers
Configuration
Theme Loader
```

## Phase 4 — ML Core

```text
Classification
Regression
Clustering
Time Series
XAI
```

## Phase 5 — ML Integration

```text
Dataset
   ↓
Preprocessing
   ↓
Training
   ↓
Evaluation
   ↓
Results
   ↓
Model Management
```

## Phase 6 — Production

```text
Security
Licensing
Optimization
Windows Build
Installer
Testing
Release
```

---

# 🔄 Example Workflow

A typical user workflow will eventually look like:

```text
Launch Tooka Desktop
        │
        ▼
     Dashboard
        │
        ▼
     New Project
        │
        ▼
    Import Dataset
        │
        ▼
   Select ML Task
        │
        ▼
 Configure Training
        │
        ▼
      Training
        │
        ▼
    Evaluation
        │
        ▼
      Results
        │
        ▼
    Save Model
```

The objective is to hide unnecessary technical complexity from the user while still providing advanced Machine Learning functionality.

---

# 💻 Running the Development Version

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m app.main
```

The exact command may change as the application architecture evolves.

---

# 📦 Future Windows Build

The final application is intended to be distributed as a Windows desktop application.

The expected distribution will eventually look similar to:

```text
Tooka Desktop
      │
      ├── Application
      ├── ML Core
      ├── Dependencies
      ├── Assets
      └── Configuration
```

A Windows executable and installer will be added during the production phase.

---

# 🔒 Security & Licensing

Because Tooka Desktop is intended to be distributed as a standalone application, security and licensing are important parts of the production phase.

Planned areas include:

* License management
* Application activation
* Secure configuration
* Model/data protection
* Dependency management
* Code packaging
* Update mechanism
* Tamper resistance

Security mechanisms will be designed separately from the Machine Learning logic so that the ML Core remains modular and maintainable.

---

# 📈 Future Improvements

Possible future features include:

* Advanced dataset explorer
* Drag-and-drop datasets
* Automatic preprocessing
* Hyperparameter optimization
* Model comparison
* Experiment tracking
* Interactive charts
* Feature importance
* Explainable AI dashboards
* Model export
* Model deployment
* Project management
* Local experiment history
* Cloud synchronization
* Automatic application updates

---

# 🤝 Development Philosophy

The project follows several architectural principles:

### Separation of Concerns

UI, application logic, and Machine Learning logic should remain independent.

### Reusability

Existing Machine Learning implementations should be reused instead of unnecessarily rewriting them.

### Modularity

Each ML task should be independently maintainable.

### Scalability

The architecture should allow new algorithms and features to be added without redesigning the entire application.

### Offline First

Core Machine Learning functionality should work locally without requiring a network connection.

### User Experience

Complex Machine Learning workflows should be presented through a simple and professional interface.

---

# 📌 Project Vision

The long-term vision of Tooka Desktop is to provide a complete local Machine Learning workspace where users can go from:

```text
Dataset
    ↓
Experiment
    ↓
Training
    ↓
Evaluation
    ↓
Explainability
    ↓
Model
```

without needing to interact directly with backend APIs or write Machine Learning code manually.

Tooka Desktop is therefore not simply a graphical interface for the existing Tooka platform.

It is being designed as a **standalone Machine Learning application built around a reusable ML Core**.

---

# 👨‍💻 Development

Tooka Desktop is currently under active development.

The project architecture, UI, ML Core, and deployment system are being developed incrementally to ensure that the final application remains maintainable, modular, and production-ready.

---

## 📄 License

License information will be added before the production release.

---

**Tooka Desktop — Making Machine Learning simpler, local, and accessible.**
