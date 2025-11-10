# 🏠 AI Space Planner

**An Intelligent Layout Generator with Setbacks, Doors & Windows**

This project uses a **Genetic Algorithm (GA)** to automatically generate **optimized architectural layouts** for small residential plans — including **Living Room, Bedroom, Kitchen, Toilet, Dining, and Balcony** — while respecting **setbacks** and automatically placing **doors and windows**.

It produces a **visual 2D floor plan** showing room placement, adjacency, and basic architectural features.

---

## 🚀 Features

- ✅ Automatically generates realistic room layouts based on area constraints
- ✅ Respects site setbacks (front, rear, left, right)
- ✅ Considers preferred adjacency between rooms (e.g., Living near Dining, Bedroom near Toilet)
- ✅ Automatically places **doors** (between adjacent rooms) and **windows** (on external walls)
- ✅ Uses **Genetic Algorithm (GA)** to evolve layouts toward higher fitness
- ✅ Visualizes the final layout using **Matplotlib**
- ✅ Simple **Gradio web interface** for interactive use

---

## 🧠 How It Works

The AI Space Planner uses **evolutionary computation** to optimize the placement of rooms within a rectangular site.

1. **Initialization** – Generates a random set of layouts (population).
2. **Fitness Evaluation** – Each layout is scored based on:
   - Room area compliance (within min-max limits)
   - Adjacency preferences (e.g., Living ↔ Dining)
   - Minimal overlap and compactness
3. **Mutation & Selection** – Slightly perturbs top-performing layouts to create new generations.
4. **Evolution** – Runs for several generations until the best layout is found.
5. **Visualization** – The final layout is drawn with labeled rooms, doors, and windows.

---

## 📦 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy matplotlib pillow gradio
```

## How to run

```bash
python run app.py
```
