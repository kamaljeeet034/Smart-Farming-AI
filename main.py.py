import customtkinter as ctk
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("Crop_recommendation.csv")

X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
y = df["label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

dt_acc = round(
    accuracy_score(y_test, dt.predict(X_test))*100,
    2
)

rf_acc = round(
    accuracy_score(y_test, rf.predict(X_test))*100,
    2
)

# ===============================
# APP
# ===============================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("1400x800")
app.configure(fg_color="#0F172A")
app.title("Smart Farming AI")

# ===============================
# HEADER
# ===============================

header = ctk.CTkFrame(
    app,
    height=100,
    fg_color="#06B16F",
    corner_radius=20
)
header.pack(fill="x",padx=10,pady=10)

ctk.CTkLabel(
    header,
    text="🌱 SMART FARMING AI DASHBOARD",
    font=("Segoe UI",18,"bold")
).pack(pady=10)

# ===============================
# CARDS
# ===============================

cards = ctk.CTkFrame(app)
cards.pack(fill="x",padx=10,pady=5)

def create_card(parent,title,value):

    frame = ctk.CTkFrame(
        parent,
        width=240,
        height=130,
        
       fg_color="#1A2740" ,
        corner_radius=25,
        border_width=2,
        border_color="#334155"
    )

    frame.pack(
        side="left",
        padx=12,
        pady=12,
        expand=True,
        fill="both"
    )
    frame.pack_propagate(False)

    ctk.CTkLabel(
        frame,
        text=title,
        font=("Segoe UI",16,"bold")
    ).pack(pady=(30,19))

    lbl = ctk.CTkLabel(
        frame,
        text=value,
        font=("Segoe UI",18,"bold")
    )

    lbl.pack(pady=(0,15))

    return lbl

    return lbl

dt_card = create_card(cards,"Decision Tree",f"{dt_acc}%")
rf_card = create_card(cards,"Random Forest",f"{rf_acc}%")
crop_card = create_card(cards,"Best Crop","---")
conf_card = create_card(cards,"Confidence","---")

# ===============================
# MAIN
# ===============================

main = ctk.CTkFrame(app)
main.pack(fill="both",expand=True,padx=10,pady=10)

left = ctk.CTkFrame(
    main,
    fg_color="#111827",
    corner_radius=25
)
left.pack(side="left",fill="both",expand=True,padx=10)

right = ctk.CTkFrame(
    main,
    width=40,
    fg_color="#16213E",
    corner_radius=25,
    border_width=2,
    border_color="#22C55E"
)
right.pack(side="right",fill="y",padx=10)

# ===============================
# INPUTS
# ===============================

input_grid = ctk.CTkFrame(
    left,
    fg_color="transparent"
)

input_grid.pack(
    fill="x",
    padx=5,
    pady=5
)

input_grid.grid_columnconfigure(0, weight=1)
input_grid.grid_columnconfigure(1, weight=1)

fields = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

entries = {}

for i, field in enumerate(fields):

    card = ctk.CTkFrame(
        input_grid,
        fg_color="#1B2A41",
        corner_radius=20
    )

    card.grid(
        row=i//2,
        column=i%2,
        padx=10,
        pady=10,
        sticky="ew"
    )

    ctk.CTkLabel(
        card,
        text=field.upper(),
        font=("Segoe UI",16,"bold")
    ).pack(anchor="w", padx=15, pady=(12,5))

    e = ctk.CTkEntry(
        card,
        height=5,
        font=("Segoe UI",18),
        corner_radius=12
    )

    e.pack(
        fill="x",
        padx=15,
        pady=(0,15)
    )

    entries[field] = e
# ===============================
# MODEL SELECTOR
# ===============================

model_var = ctk.StringVar(value="Random Forest")

model_box = ctk.CTkComboBox(
    left,
    values=[
        "Random Forest",
        "Decision Tree"
    ],
    variable=model_var
)

model_box.pack(pady=11)

# ===============================
# RESULT AREA
# ===============================

crop_label = ctk.CTkLabel(
    right,
    text="🌱",
    font=("Segoe UI Emoji",100)
)

crop_label.pack(pady=20)

result_label = ctk.CTkLabel(
    right,
    text="No Prediction",
    font=("Segoe UI",28,"bold"),
    text_color="#22C55E"
)

result_label.pack()

info_label = ctk.CTkLabel(
    right,
    text="",
    wraplength=350
)

info_label.pack(pady=10)

top3_box = ctk.CTkTextbox(
    right,
    width=250,
    height=20,
    corner_radius=15,
    font=("Consolas",16)
)

top3_box.pack(pady=20)

# ===============================
# PREDICT
# ===============================

def predict():

    try:

        vals = [
            float(entries["N"].get()),
            float(entries["P"].get()),
            float(entries["K"].get()),
            float(entries["temperature"].get()),
            float(entries["humidity"].get()),
            float(entries["ph"].get()),
            float(entries["rainfall"].get())
        ]

        arr = np.array([vals])

        if model_var.get() == "Decision Tree":

            pred = dt.predict(arr)[0]
            prob = dt.predict_proba(arr).max()

        else:

            pred = rf.predict(arr)[0]
            prob = rf.predict_proba(arr).max()

        crop = le.inverse_transform([pred])[0]

        probs = rf.predict_proba(arr)[0]
        idx = probs.argsort()[-3:][::-1]

        top3_box.delete("1.0","end")

        for i in idx:

            name = le.inverse_transform([i])[0]

            top3_box.insert(
                "end",
                f"{name} - {round(probs[i]*100,1)}%\n"
            )

        result_label.configure(
            text=crop.upper()
        )

        crop_card.configure(
            text=crop
        )

        conf_card.configure(
            text=f"{round(prob*100,1)}%"
        )

    except Exception as e:

        result_label.configure(
            text=f"Error: {e}"
        )

predict_btn = ctk.CTkButton(
    left,
    text="🌱 PREDICT CROP",
    width=60,
    height=15,
    corner_radius=20,
    fg_color="#0B383D",
    hover_color="#16A34A",
    font=("Segoe UI",20,"bold"),
    command=predict
)

predict_btn.pack(pady=0)

app.mainloop()