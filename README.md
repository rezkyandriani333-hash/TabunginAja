# Tabungin — Student Financial Planner

File di repo:
- app.py
- requirements.txt
- README.md

Cara deploy (Streamlit Community Cloud):
1. Pastikan file app.py & requirements.txt ada di root repository.
2. Commit & push ke GitHub.
3. Buka https://streamlit.io/cloud → New app → pilih repo → Main file path: app.py → Deploy.
4. Jika memakai AI, tambahkan secret `GROQ_API_KEY` di Settings → Secrets.

Menjalankan lokal:
1. pip install -r requirements.txt
2. streamlit run app.py
