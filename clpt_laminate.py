import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

# =========================================================
# UI TUNING
# =========================================================
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h2, h3 {margin-bottom: 0.3rem;}

/* Reduce metric font size */
div[data-testid="stMetricValue"] {
    font-size: 16px !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 1e-8

# =========================================================
# MATERIAL DATABASE
# =========================================================
MATERIAL_DB = {
    "Carbon/Epoxy": (135.000, 10.000, 5.000, 0.300),
    "Glass/Epoxy": (40.000, 10.000, 4.000, 0.280),
    "Kevlar/Epoxy": (70.000, 5.000, 2.500, 0.340)
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

    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))

    for k in range(len(layers)):
        Qb = transform_Q(get_Q(
            layers[k]['E1'],
            layers[k]['E2'],
            layers[k]['G12'],
            layers[k]['nu12']),
            layers[k]['theta'])

        z1,z2 = z[k],z[k+1]

        A += Qb*(z2-z1)
        B += 0.5*Qb*(z2**2-z1**2)
        D += (1/3)*Qb*(z2**3-z1**3)

    return clean(A),clean(B),clean(D),h

def engineering_constants_from_Q(Qbar):
    S = np.linalg.inv(Qbar)
    return (
        1/S[0,0],
        1/S[1,1],
        1/S[2,2],
        -S[0,1]/S[0,0]
    )

def build_symmetric_laminate(layers, symmetric):
    return layers + [l.copy() for l in reversed(layers)] if symmetric else layers

def check_symmetry(B, tol=1e-6):
    return np.all(np.abs(B) < tol)

def laminate_notation(layers, symmetric, n):
    seq = [str(int(layers[i]['theta'])) for i in range(n)]
    return f"[{'/'.join(seq)}]s" if symmetric else f"[{'/'.join(seq)}]"

# =========================================================
# LAYOUT
# =========================================================
colL, colM, colR = st.columns([1,2,2])

# =========================================================
# MATERIAL PANEL
# =========================================================
with colL:
    st.markdown("## Material")

    mat_choice = st.selectbox("Select Material", ["Custom"] + list(MATERIAL_DB.keys()))

    if mat_choice != "Custom":
        E1_def, E2_def, G12_def, nu12_def = MATERIAL_DB[mat_choice]
    else:
        E1_def, E2_def, G12_def, nu12_def = 25.000, 1.000, 0.500, 0.270

    E1 = st.number_input("E₁ (GPa)", value=E1_def, format="%.3f")
    E2 = st.number_input("E₂ (GPa)", value=E2_def, format="%.3f")
    G12 = st.number_input("G₁₂ (GPa)", value=G12_def, format="%.3f")
    nu12 = st.number_input("ν₁₂", value=nu12_def, format="%.3f")

# =========================================================
# STACKING
# =========================================================
with colM:
    st.markdown("## Laminate")

    cA,cB,cC = st.columns(3)
    n = cA.number_input("Layers",1,20,4)
    symmetric = cB.checkbox("Symmetric", True)
    same_t = cC.checkbox("same thickness", True)

    st.markdown("### Stacking Sequence (Top → Bottom)")

    base_t = st.number_input("Thickness (mm)", value=0.100, format="%.3f")


    # HEADER
    h1,h2,h3 = st.columns([1,1,1])
    h1.markdown("**Ply**")
    h2.markdown("**t (mm)**")
    h3.markdown("**θ (deg)**")

    subs = ["₁","₂","₃","₄","₅","₆","₇","₈","₉","₁₀"]

    layers=[]
    for i in range(n):
        c1,c2,c3 = st.columns([1,1,1])
        c1.write(i+1)

        t = base_t if same_t else c2.number_input(f"t{i}",0.100,key=f"t{i}")
        if same_t: c2.write(f"{t:.3f}")

        theta = c3.number_input(
            f"θ{subs[i]}",
            min_value=-360.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
            key=f"th{i}"
        )

        layers.append({
            "E1":E1*1e9,"E2":E2*1e9,"G12":G12*1e9,
            "nu12":nu12,"theta":theta,"t":t*1e-3
        })

    layers = build_symmetric_laminate(layers, symmetric)

    total_t = sum(l['t'] for l in layers)
    notation = laminate_notation(layers, symmetric, n)

    st.markdown("---")
    c1,c2 = st.columns(2)
    c1.markdown(f"**Laminate:** `{notation}`")
    c2.markdown(f"**Plies:** {len(layers)}")

    st.markdown(f"**Total Thickness:** {total_t*1000:.3f} mm")

# =========================================================
# COMPUTE
# =========================================================
A,B,D,h = compute_ABD(layers)
is_sym = check_symmetry(B)

Qbar_mem = A / h
Qbar_flex = (12 / (h**3)) * D

Ex,Ey,Gxy,nuxy = engineering_constants_from_Q(Qbar_mem)

if is_sym:
    Exf,Eyf,Gxyf,_ = engineering_constants_from_Q(Qbar_flex)
else:
    Exf=Eyf=Gxyf=None

# =========================================================
# RESULTS
# =========================================================
with colR:
    st.markdown("## Results")

    unit = st.radio("Units", ["N/mm","N/m"], horizontal=True)

    if is_sym:
        st.success("Symmetric laminate")
    else:
        st.error("Unsymmetric laminate (B ≠ 0)")

    st.markdown("### Membrane")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Eₓ(GPa)", f"{Ex/1e9:.2f}")
    c2.metric("Eᵧ(GPa)", f"{Ey/1e9:.2f}")
    c3.metric("Gₓᵧ(GPa)", f"{Gxy/1e9:.2f}")
    c4.metric("νₓᵧ", f"{nuxy:.3f}")

    st.markdown("### Flexural")

    if is_sym:
        c5,c6,c7 = st.columns(3)
        c5.metric("Eₓᶠ(GPa)", f"{Exf/1e9:.2f}")
        c6.metric("Eᵧᶠ(GPa)", f"{Eyf/1e9:.2f}")
        c7.metric("Gₓᵧᶠ(GPa)", f"{Gxyf/1e9:.2f}")
    else:
        st.warning("Not valid for unsymmetric laminate")

    st.markdown("### ABD Matrices")

    def scale(name,M):
        if unit=="N/mm":
            return (M/1000,"N/mm") if name=="A" else \
                   (M,"N") if name=="B" else \
                   (M*1000,"N·mm")
        else:
            return (M,"N/m") if name=="A" else \
                   (M,"N") if name=="B" else \
                   (M,"N·m")

    idx_map = ["1","2","6"]

    for k,(name,M) in enumerate(zip(["A","B","D"],[A,B,D])):

        if k > 0:
            st.markdown("---")   # separator line

        Mscaled,u = scale(name,M)
        st.markdown(f"**{name} ({u})**")

        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                label = f"{name}{idx_map[i]}{idx_map[j]}"
                value = f"{Mscaled[i,j]:.2f}"
                cols[j].metric(label, value)
