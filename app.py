# Tabungin — Student Financial Planner (app.py)
# Penjelasan: Aplikasi Streamlit untuk mahasiswa.
# Fitur utama: upload CSV/XLSX, input manual, ringkasan bulanan,
# Dashboard C: Tabungan & Proyeksi, goals, dan AI insights opsional via Groq.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# plotting
import plotly.express as px
import plotly.graph_objects as go

# Optional Groq client
try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

APP_NAME = "Tabungin — Student Financial Planner"

st.set_page_config(page_title=APP_NAME, page_icon="💸", layout="wide")
st.title(f"💸 {APP_NAME}")
st.write("Aplikasi untuk mencatat pemasukan & pengeluaran, memonitor tabungan, dan mensimulasikan target 3–6 bulan. Fitur AI opsional (Groq).")

# Sidebar - settings / upload / manual entry
st.sidebar.header("Upload / Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah file transaksi (CSV atau Excel)", type=["csv", "xlsx", "xls"])
st.sidebar.markdown("**Format minimal file:** date, type (Income/Expense), category, amount")

selected_model = st.sidebar.selectbox("Pilih model AI (jika tersedia)", 
                                      ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"])

st.sidebar.markdown("---")
st.sidebar.subheader("Tambah transaksi manual")
with st.sidebar.form(key="manual_form"):
    date_in = st.date_input("Tanggal", value=datetime.today())
    type_in = st.selectbox("Tipe", ["Income", "Expense"])
    category_in = st.text_input("Kategori", value="Lain-lain")
    amount_in = st.number_input("Jumlah (positif, IDR)", min_value=0.0, value=0.0, format="%.2f")
    add_txn = st.form_submit_button("Tambah transaksi")

# session state: transactions and goals and chat history
if "df_txn" not in st.session_state:
    st.session_state.df_txn = pd.DataFrame(columns=["date", "type", "category", "amount"])
if "goals" not in st.session_state:
    st.session_state.goals = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process uploaded file
def normalize_and_append(df_raw):
    df = df_raw.copy()
    # standardize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # mapping heuristics
    mapping = {}
    for col in df.columns:
        if "date" in col:
            mapping["date"] = col
        if col in ["type", "tipe"] or "income" in col or "expense" in col:
            if "type" not in mapping:
                mapping["type"] = col
        if "category" in col or "kategori" in col:
            mapping["category"] = col
        if "amount" in col or "jumlah" in col or "nominal" in col or "value" in col or "amount (id)" in col:
            mapping["amount"] = col
    required = ["date", "type", "category", "amount"]
    if not all(k in mapping for k in required):
        return None, f"File tidak memiliki kolom minimal: date, type, category, amount. Ditemukan: {list(df.columns)}"
    df = df[[mapping["date"], mapping["type"], mapping["category"], mapping["amount"]]]
    df.columns = ["date", "type", "category", "amount"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["type"] = df["type"].astype(str).apply(lambda s: "Income" if "inc" in s.lower() else ("Expense" if "exp" in s.lower() else s))
    return df, None

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        df_proc, err = normalize_and_append(df_raw)
        if err:
            st.sidebar.error(err)
        else:
            st.session_state.df_txn = pd.concat([st.session_state.df_txn, df_proc], ignore_index=True)
            st.sidebar.success("File berhasil diunggah.")
    except Exception as e:
        st.sidebar.error(f"Gagal memproses file: {e}")

# Add manual txn
if add_txn and amount_in > 0:
    new = pd.DataFrame([{
        "date": pd.to_datetime(date_in),
        "type": type_in,
        "category": category_in,
        "amount": float(amount_in)
    }])
    st.session_state.df_txn = pd.concat([st.session_state.df_txn, new], ignore_index=True)
    st.sidebar.success("Transaksi manual ditambahkan.")

# If no data: show template & stop
if st.session_state.df_txn.empty:
    st.info("Belum ada data transaksi — unggah file atau tambahkan transaksi manual di sidebar. Berikut contoh dataset.")
    sample = pd.DataFrame([
        {"date": "2025-01-05", "type": "Income", "category": "Uang Saku", "amount": 1000000},
        {"date": "2025-01-06", "type": "Expense", "category": "Makan", "amount": 25000},
        {"date": "2025-01-10", "type": "Expense", "category": "Transport", "amount": 15000},
    ])
    st.download_button("Download contoh CSV", sample.to_csv(index=False).encode("utf-8"), file_name="template_transactions.csv", mime="text/csv")
    st.stop()

# Prepare dataframe
df = st.session_state.df_txn.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["type"] = df["type"].astype(str)
df["category"] = df["category"].astype(str)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

# Monthly aggregation
monthly = df.groupby(["month", "type"]).agg(total=("amount", "sum")).reset_index()
monthly_pivot = monthly.pivot(index="month", columns="type", values="total").fillna(0.0)
monthly_pivot = monthly_pivot.sort_index()
monthly_pivot["Income"] = monthly_pivot.get("Income", 0.0)
monthly_pivot["Expense"] = monthly_pivot.get("Expense", 0.0)
monthly_pivot["Net"] = monthly_pivot["Income"] - monthly_pivot["Expense"]
monthly_pivot = monthly_pivot.reset_index()
monthly_pivot["Cumulative Savings"] = monthly_pivot["Net"].cumsum()
monthly_pivot["Savings Rate"] = monthly_pivot.apply(lambda r: (r["Net"]/r["Income"]*100) if r["Income"]>0 else 0.0, axis=1)

# Layout: main + side
col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📋 Daftar Transaksi (terbaru)")
    st.dataframe(df.sort_values("date", ascending=False).reset_index(drop=True))

    st.subheader("📈 Ringkasan Bulanan")
    st.dataframe(monthly_pivot[["month", "Income", "Expense", "Net", "Cumulative Savings", "Savings Rate"]].fillna(0).round(2))

    fig_ie = px.bar(monthly_pivot, x="month", y=["Income", "Expense"], barmode="group", title="Pemasukan vs Pengeluaran per Bulan")
    st.plotly_chart(fig_ie, use_container_width=True)

with col_side:
    st.subheader("⚙️ Financial Goals & Statistik")
    avg_income = monthly_pivot["Income"].replace(0, np.nan).mean()
    st.metric("Rata-rata Pemasukan Bulanan", f"{avg_income:,.0f}" if not np.isnan(avg_income) else "-")
    # Goals form
    with st.form("goal_form"):
        goal_name = st.text_input("Nama Goal (contoh: Laptop)")
        goal_target = st.number_input("Jumlah Target (IDR)", min_value=0.0, value=0.0, step=10000.0, format="%.2f")
        goal_saved = st.number_input("Tersimpan Saat Ini (IDR)", min_value=0.0, value=0.0, step=10000.0, format="%.2f")
        save_goal = st.form_submit_button("Simpan Goal")
    if save_goal and goal_name and goal_target > 0:
        st.session_state.goals.append({"name": goal_name, "target": goal_target, "saved": goal_saved})
        st.success("Goal tersimpan.")

    if st.session_state.goals:
        for g in st.session_state.goals:
            prog = min(1.0, g["saved"]/g["target"]) if g["target"]>0 else 0.0
            st.write(f"**{g['name']}** — Target: {g['target']:,.0f} | Tersimpan: {g['saved']:,.0f}")
            st.progress(prog)

# Dashboard C — Tabungan & Proyeksi
st.markdown("---")
st.header("💰 Dashboard C — Tabungan & Potensi Menabung")
colA, colB = st.columns([2, 1])

with colA:
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(x=monthly_pivot["month"], y=monthly_pivot["Cumulative Savings"], fill="tozeroy", name="Cumulative Savings"))
    fig_area.update_layout(title="Akumulasi Tabungan (Cumulative Savings)", xaxis_title="Bulan", yaxis_title="IDR")
    st.plotly_chart(fig_area, use_container_width=True)

    fig_bar = px.bar(monthly_pivot, x="month", y="Savings Rate", title="Savings Rate (%) per Bulan")
    st.plotly_chart(fig_bar, use_container_width=True)

with colB:
    st.subheader("🔮 Simulasi Target 3–6 Bulan")
    sim_target = st.number_input("Target Tabungan (IDR) untuk simulasi", min_value=0.0, value=1000000.0, step=10000.0, format="%.2f")
    avg_positive = monthly_pivot[monthly_pivot["Net"]>0]["Net"].mean()
    st.write(f"Rata-rata Net per bulan (semua): {monthly_pivot['Net'].mean():,.0f}")
    st.write(f"Rata-rata Net (bulan positif): {avg_positive:,.0f}" if not np.isnan(avg_positive) else "Tidak ada bulan positif")

    def simulate_months(target, current_saved, monthly_save):
        if monthly_save <= 0:
            return None
        months = 0
        saved = current_saved
        while saved < target and months < 1200:
            saved += monthly_save
            months += 1
        return months

    current_saved_total = monthly_pivot["Cumulative Savings"].iloc[-1] if len(monthly_pivot)>0 else 0.0
    months_needed = simulate_months(sim_target, current_saved_total, avg_positive if not np.isnan(avg_positive) else 0)
    if months_needed is None:
        st.warning("Tidak cukup surplus saat ini untuk mencapai target—pertimbangkan mengurangi pengeluaran atau tambah pemasukan.")
    else:
        st.success(f"Estimasi waktu mencapai target (dengan rata-rata bulan positif): {months_needed} bulan")

    if avg_positive > 0:
        plan3 = max(0, (sim_target - current_saved_total)/3)
        plan6 = max(0, (sim_target - current_saved_total)/6)
        st.write(f"Perlu sisihkan ~{plan3:,.0f}/bulan untuk target 3 bulan, atau {plan6:,.0f}/bulan untuk target 6 bulan.")
    else:
        st.write("Tidak ada rekomendasi angka bulanan karena rata-rata surplus ≤ 0.")

# AI Insights (optional)
st.markdown("---")
st.header("🤖 AI Insights (Opsional)")

if not HAS_GROQ or not GROQ_API_KEY:
    st.info("Fitur AI tidak aktif — pasang paket 'groq' dan set GROQ_API_KEY untuk mengaktifkan.")
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        preview = df.sort_values("date", ascending=False).head(40).to_string(index=False)
        prompt = f"Kamu adalah penasihat keuangan untuk mahasiswa. Berikut ringkasan transaksi (preview):\n{preview}\n\nTuliskan 3 langkah cepat untuk meningkatkan tabungan dan rekomendasi strategi 3 & 6 bulan (poin-poin singkat)."
        if st.button("🧠 Minta AI Insight"):
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful financial coach for students. Keep answers concise and practical."},
                        {"role": "user", "content": prompt}
                    ],
                    model=selected_model,
                )
                st.subheader("AI Insight")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"Permintaan AI gagal: {e}")
    except Exception as e:
        st.error(f"Gagal inisialisasi Groq client: {e}")

# Simple chat (persist to session)
st.markdown("---")
st.subheader("💬 Chat Perencanaan (opsional)")
q = st.text_input("Tanyakan sesuatu terkait keuanganmu (mis: Cara menabung untuk laptop 6 bulan?)")
colc1, colc2 = st.columns([1,1])
with colc1:
    send = st.button("Send")
with colc2:
    clear = st.button("Clear Chat")
if clear:
    st.session_state.chat_history = []
if send and q:
    # local heuristic reply if no AI
    if not HAS_GROQ or not GROQ_API_KEY:
        reply = "Saran singkat: (1) Buat anggaran bulanan; (2) kurangi pengeluaran variabel; (3) sisihkan nominal tetap setiap minggu; (4) cari pemasukan tambahan."
        st.session_state.chat_history.append((q, reply))
    else:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            messages = [
                {"role": "system", "content": "You are a helpful financial assistant for students. Keep answers short and actionable."},
                {"role": "user", "content": f"Context (last months):\n{monthly_pivot[['month','Income','Expense','Net']].tail(6).to_string(index=False)}\n\nUser question: {q}"}
            ]
            chatresp = client.chat.completions.create(messages=messages, model=selected_model)
            ans = chatresp.choices[0].message.content
            st.session_state.chat_history.append((q, ans))
        except Exception as e:
            st.error(f"Chat AI gagal: {e}")

if st.session_state.chat_history:
    for user_q, ans in reversed(st.session_state.chat_history):
        st.markdown(f"**👤 You:** {user_q}")
        st.markdown(f"**🤖 Copilot:** {ans}")

# Footer / instructions
st.markdown("---")
st.write("Petunjuk singkat: pastikan file memiliki kolom: date,type,category,amount. Untuk fitur AI, set GROQ_API_KEY sebagai secret di Streamlit Cloud atau di .env (lokal).")
