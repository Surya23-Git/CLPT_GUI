import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

# =========================================================
# COMPACT UI
# =========================================================
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
h2, h3 {margin-bottom: 0.3rem;}
label {font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 1e-8

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

def engineering_constants(A,h):
    S=np.linalg.inv(A)
    return (
        1/(h*S[0,0]),
        1/(h*S[1,1]),
        1/(h*S[2,2]),
        -S[0,1]/S[0,0]
    )

def build_symmetric_laminate(layers, symmetric):
    return layers + [l.copy() for l in reversed(layers)] if symmetric else layers

def check_symmetry(B, tol=1e-6):
    return np.all(np.abs(B) < tol)

# =========================================================
# LAYOUT
# =========================================================
colL, colM, colR = st.columns([1,2,2])

# =========================================================
# MATERIAL
# =========================================================
with colL:
    st.markdown("## Material Properties")

    E1 = st.number_input("E₁ (GPa)", value=150.0)
    E2 = st.number_input("E₂ (GPa)", value=20.0)
    G12 = st.number_input("G₁₂ (GPa)", value=5.0)
    nu12 = st.number_input("ν₁₂", value=0.27)

# =========================================================
# STACKING
# =========================================================
with colM:
    st.markdown("## Laminate Definition")

    cA,cB,cC = st.columns(3)
    num_layers = cA.number_input("Layers",1,20,4)
    symmetric = cB.checkbox("Symmetric", True)
    same_t = cC.checkbox("Same thickness", True)

    st.markdown("### Stacking Sequence (Top → Mid-plane)")

    base_t = st.number_input("Thickness (mm)", value=0.15)

    layers=[]

    for i in range(num_layers):
        c1,c2 = st.columns([1,1])

        # Thickness
        if same_t:
            c1.write(f"t{i+1} = {base_t:.3f}")
            t = base_t
        else:
            t = c1.number_input(f"t{i+1}", value=0.15, key=f"t{i}")

        # Angle with subscript
        theta = c2.number_input(
            f"θ_{i+1} (deg)",
            value=0.0,
            key=f"th{i}"
        )

        layers.append({
            "E1":E1*1e9,"E2":E2*1e9,"G12":G12*1e9,
            "nu12":nu12,"theta":theta,"t":t*1e-3
        })

    layers = build_symmetric_laminate(layers, symmetric)

    total_t = sum(l['t'] for l in layers)
    st.caption(f"{len(layers)} plies | {total_t*1000:.2f} mm")

# =========================================================
# COMPUTE
# =========================================================
A,B,D,h = compute_ABD(layers)
is_sym = check_symmetry(B)

Ex,Ey,Gxy,nuxy = engineering_constants(A,h)
Exf,Eyf,Gxyf,_ = engineering_constants(D,h) if is_sym else (None,None,None,None)

# =========================================================
# RESULTS
# =========================================================
with colR:

    st.markdown("## Results")

    unit = st.radio("Units", ["N/mm","N/m"], horizontal=True)

    # STATUS
    if is_sym:
        st.success("Symmetric laminate (B ≈ 0)")
    else:
        st.error("Unsymmetric laminate (B ≠ 0)")
        st.warning("Extension-bending coupling present")

    # -----------------------------
    # ENGINEERING CONSTANTS
    # -----------------------------
    st.markdown("### Membrane Constants")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Eₓ", f"{Ex/1e9:.2f} GPa")
    c2.metric("Eᵧ", f"{Ey/1e9:.2f} GPa")
    c3.metric("Gₓᵧ", f"{Gxy/1e9:.2f} GPa")
    c4.metric("νₓᵧ", f"{nuxy:.3f}")

    st.markdown("### Flexural Constants")

    if is_sym:
        c5,c6,c7 = st.columns(3)
        c5.metric("Eₓᶠ", f"{Exf/1e9:.2f} GPa")
        c6.metric("Eᵧᶠ", f"{Eyf/1e9:.2f} GPa")
        c7.metric("Gₓᵧᶠ", f"{Gxyf/1e9:.2f} GPa")
    else:
        st.info("Flexural constants not valid for unsymmetric laminates")

    # -----------------------------
    # ABD
    # -----------------------------
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

    for name,M in zip(["A","B","D"],[A,B,D]):
        Mscaled,u = scale(name,M)
        st.markdown(f"**{name} ({u})**")

        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                cols[j].write(f"{Mscaled[i,j]:.2f}")

    # -----------------------------
    # EXPORT
    # -----------------------------
    st.markdown("### Export")

    ABD = np.block([[A,B],[B,D]])

    st.download_button(
        "Download CSV",
        pd.DataFrame(ABD).to_csv().encode(),
        "ABD.csv"
    )

    def to_latex(M,name):
        return name+" = \\begin{bmatrix}\n"+\
               "\\\\\n".join([" & ".join([f"{v:.2e}" for v in r]) for r in M])+\
               "\n\\end{bmatrix}"

    latex = to_latex(A,"A")+"\n\n"+to_latex(B,"B")+"\n\n"+to_latex(D,"D")

    st.download_button("Download LaTeX", latex, "ABD.tex")
