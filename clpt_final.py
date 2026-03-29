import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

THRESHOLD = 1e-5

# =========================================================
# MATERIAL LIBRARY
# =========================================================
MATERIAL_DB = {
    "Carbon/Epoxy": {"E1":135, "E2":10, "G12":5, "nu12":0.3},
    "Glass/Epoxy": {"E1":40, "E2":10, "G12":4, "nu12":0.28},
    "Kevlar/Epoxy": {"E1":70, "E2":5, "G12":2.5, "nu12":0.34}
}

# =========================================================
# FUNCTIONS
# =========================================================
def clean(M):
    M[np.abs(M) < THRESHOLD] = 0
    return M

def get_Q(E1,E2,G12,nu12):
    nu21 = (E2/E1)*nu12
    d = 1 - nu12*nu21
    return np.array([[E1/d, nu12*E2/d, 0],
                     [nu12*E2/d, E2/d, 0],
                     [0,0,G12]])

def transform_Q(Q,theta):
    t = np.radians(theta)
    m,n = np.cos(t), np.sin(t)
    T = np.array([[m*m,n*n,2*m*n],
                  [n*n,m*m,-2*m*n],
                  [-m*n,m*n,m*m-n*n]])
    Ti = np.linalg.inv(T)
    return Ti @ Q @ Ti.T

def compute_ABD(layers):
    h = sum(l['t'] for l in layers)
    z=[-h/2]
    for l in layers:
        z.append(z[-1]+l['t'])

    A,B,D = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))

    for k in range(len(layers)):
        Qb = transform_Q(get_Q(layers[k]['E1'],layers[k]['E2'],
                               layers[k]['G12'],layers[k]['nu12']),
                         layers[k]['theta'])
        z1,z2 = z[k],z[k+1]

        A += Qb*(z2-z1)
        B += 0.5*Qb*(z2**2 - z1**2)
        D += (1/3)*Qb*(z2**3 - z1**3)

    return clean(A),clean(B),clean(D),h

def engineering_constants(A,h):
    S=np.linalg.inv(A)
    Ex=1/(h*S[0,0])
    Ey=1/(h*S[1,1])
    Gxy=1/(h*S[2,2])
    nuxy=-S[0,1]/S[0,0]
    return Ex,Ey,Gxy,nuxy

def plot3D(layers):
    z=0
    fig=go.Figure()

    for l in layers:
        t=l['t']; th=l['theta']
        x=[0,1,1,0,0,1,1,0]
        y=[0,0,1,1,0,0,1,1]
        zc=[z,z,z,z,z+t,z+t,z+t,z+t]

        color=f"hsl({(th%180)*2},70%,50%)"

        fig.add_trace(go.Mesh3d(x=x,y=y,z=zc,color=color,opacity=0.85))
        fig.add_trace(go.Cone(
            x=[0.5],y=[0.5],z=[z+t/2],
            u=[np.cos(np.radians(th))],
            v=[np.sin(np.radians(th))],
            w=[0],showscale=False
        ))

        z+=t

    fig.update_layout(height=400,margin=dict(l=0,r=0,t=20,b=0))
    return fig

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🧩 Model Setup")

num_layers = st.sidebar.number_input("Layers",1,20,4)
same_t = st.sidebar.checkbox("Same Thickness")
same_m = st.sidebar.checkbox("Same Material")

mat_choice = st.sidebar.selectbox("Material", ["Custom"] + list(MATERIAL_DB.keys()))

if mat_choice != "Custom":
    mat = MATERIAL_DB[mat_choice]

if same_t:
    t_global = st.sidebar.number_input(
        "t (mm)", min_value=1e-6, value=1.0, step=0.1, format="%.4f"
    )

if same_m:
    if mat_choice != "Custom":
        E1g,E2g,G12g,nu12g = mat["E1"],mat["E2"],mat["G12"],mat["nu12"]
    else:
        E1g = st.sidebar.number_input("E1",min_value=0.0,value=25.0)
        E2g = st.sidebar.number_input("E2",min_value=0.0,value=1.0)
        G12g = st.sidebar.number_input("G12",min_value=0.0,value=0.5)
        nu12g = st.sidebar.number_input("nu12",value=0.25)

# =========================================================
# RESET
# =========================================================
if st.sidebar.button("Reset Table"):
    st.session_state.df = None
    st.rerun()

# =========================================================
# SESSION INIT
# =========================================================
if "df" not in st.session_state or st.session_state.df is None or len(st.session_state.df)!=num_layers:
    st.session_state.df = pd.DataFrame({
        "θ":[0]*num_layers,
        "t":[1.0]*num_layers,
        "E1":[25.0]*num_layers,
        "E2":[1.0]*num_layers,
        "G12":[0.5]*num_layers,
        "ν12":[0.25]*num_layers
    })

# =========================================================
# APPLY GLOBAL
# =========================================================
if same_t:
    st.session_state.df["t"] = t_global

if same_m:
    st.session_state.df[["E1","E2","G12","ν12"]] = [E1g,E2g,G12g,nu12g]

# =========================================================
# TABLE (FIXED FOR NEGATIVE ANGLES)
# =========================================================
disabled_cols = ["E1","E2","G12","ν12"] if same_m else []

df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    disabled=disabled_cols,
    column_config={
        "θ": st.column_config.NumberColumn(
            "θ (deg)",
            min_value=-360.0,
            max_value=360.0,
            step=1.0
        ),
        "t": st.column_config.NumberColumn(
            "t (mm)",
            min_value=1e-6,
            step=0.1
        ),
        "E1": st.column_config.NumberColumn("E1 (GPa)", min_value=0.0),
        "E2": st.column_config.NumberColumn("E2 (GPa)", min_value=0.0),
        "G12": st.column_config.NumberColumn("G12 (GPa)", min_value=0.0),
        "ν12": st.column_config.NumberColumn("ν12")
    }
)

st.session_state.df = df

# =========================================================
# VALIDATION
# =========================================================
errors = []

for i, row in df.iterrows():
    E1,E2,G12,nu12,t = row["E1"],row["E2"],row["G12"],row["ν12"],row["t"]

    if t <= 0:
        errors.append(f"Layer {i+1}: Thickness must be > 0")

    if E1 <= 0 or E2 <= 0 or G12 <= 0:
        errors.append(f"Layer {i+1}: Elastic constants must be positive")

    if E2 > E1:
        errors.append(f"Layer {i+1}: E2 should be ≤ E1")

    nu21 = (E2/E1)*nu12
    if nu12*nu21 >= 1:
        errors.append(f"Layer {i+1}: Stability violated")

if errors:
    st.error("Invalid Inputs:")
    for e in errors:
        st.write(e)
    st.stop()

# =========================================================
# COMPUTE
# =========================================================
layers=[{
    'E1':r.E1*1e9,'E2':r.E2*1e9,'G12':r.G12*1e9,
    'nu12':r["ν12"],'theta':r["θ"],'t':r.t*1e-3
} for _,r in df.iterrows()]

A,B,D,h = compute_ABD(layers)
Ex,Ey,Gxy,nuxy = engineering_constants(A,h)

# =========================================================
# OUTPUT TOP
# =========================================================
col1,col2 = st.columns([1,1])

with col1:
    st.subheader("Effective Properties")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ex (GPa)",f"{Ex/1e9:.2f}")
    c2.metric("Ey (GPa)",f"{Ey/1e9:.2f}")
    c3.metric("Gxy (GPa)",f"{Gxy/1e9:.2f}")
    c4.metric("νxy",f"{nuxy:.3f}")

with col2:
    st.subheader("3D Laminate")
    st.plotly_chart(plot3D(layers),use_container_width=True)

# =========================================================
# ABD OUTPUT
# =========================================================
st.subheader("ABD Matrices")

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("**A Matrix (N/mm)**")
    st.dataframe(np.round(A/1000,3))

with c2:
    st.markdown("**B Matrix (N)**")
    st.dataframe(np.round(B,3))

with c3:
    st.markdown("**D Matrix (N·mm)**")
    st.dataframe(np.round(D*1000,3))
