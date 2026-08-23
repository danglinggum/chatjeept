from __future__ import annotations

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

from rag import init_and_populate_vector_db

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY is missing.")
    sys.exit(1)

ai_client = genai.Client(api_key=GOOGLE_API_KEY)

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = Path(__file__).resolve().parent / "raw_papers"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 과목별 표준 파일명 매핑
SUBJECT_FILE_MAP = {
    "Physics": "physics.json",
    "Chemistry": "chemistry.json",
    "Mathematics": "math.json"
}

# =============================================================================
# 1. 고품질 JEE 시드 벤치마크 데이터셋
# =============================================================================

SEED_DATA: Dict[str, List[Dict[str, Any]]] = {
    "Physics": [
        {
            "id": "PHY-2023-ADV-P1-01",
            "exam": "JEE Advanced 2023",
            "topic": "Rotational Mechanics & Pure Rolling",
            "difficulty": "Hard",
            "question": r"A uniform solid cylinder of mass $M$ and radius $R$ is placed on a rough horizontal floor with coefficient of friction $\mu$. A constant horizontal force $F$ is applied tangentially at the top of the cylinder. Determine the linear acceleration $a_{cm}$ and the minimum coefficient of friction $\mu_{min}$ for rolling without slipping.",
            "solution": r"1. Dynamics: Force equation $F + f = M a_{cm}$. Torque about CM: $(F - f)R = I_{cm}\alpha = \left(\frac{1}{2}MR^2\right)\alpha$." + "\n" + r"2. Pure rolling: $a_{cm} = R\alpha \implies F - f = \frac{1}{2} M a_{cm}$." + "\n" + r"3. Solving gives $a_{cm} = \frac{4F}{3M}$ and static friction $f = \frac{F}{3}$." + "\n" + r"4. For no slipping: $f \le \mu N = \mu Mg \implies \mu_{min} = \frac{F}{3Mg}$.",
            "visual_tags": ["cylinder", "force_vector", "friction_vector"]
        },
        {
            "id": "PHY-2022-ADV-P2-04",
            "exam": "JEE Advanced 2022",
            "topic": "Electromagnetism & Helical Motion",
            "difficulty": "Hard",
            "question": r"A particle of mass $m$ and charge $+q$ is projected with velocity $\vec{v} = v_0 \cos\theta\hat{i} + v_0 \sin\theta\hat{j}$ into a uniform magnetic field $\vec{B} = B_0 \hat{i}$. Find the radius of the helix, pitch of trajectory, and magnetic force vector.",
            "solution": r"1. Parallel component $v_\parallel = v_0\cos\theta$ causes linear motion along x-axis with pitch $p = v_\parallel T = (v_0\cos\theta)\left(\frac{2\pi m}{qB_0}\right)$." + "\n" + r"2. Perpendicular component $v_\perp = v_0\sin\theta$ generates circular motion with radius $R = \frac{m v_0\sin\theta}{qB_0}$." + "\n" + r"3. Lorentz force: $\vec{F} = q(\vec{v}\times\vec{B}) = -q v_0 B_0 \sin\theta \hat{k}$.",
            "visual_tags": ["helical_path", "velocity_components", "magnetic_force_vector"]
        },
        {
            "id": "PHY-2023-MAIN-JAN-08",
            "exam": "JEE Main 2023",
            "topic": "Electrostatics & Electric Dipole",
            "difficulty": "Medium",
            "question": r"An electric dipole of moment $\vec{p} = p_0\hat{i}$ is placed at the origin in an electric field $\vec{E} = E_0(x\hat{i} + y\hat{j})$. Compute the net electrostatic force and torque acting on the dipole.",
            "solution": r"1. Torque: $\vec{\tau} = \vec{p} \times \vec{E}(0) = (p_0\hat{i}) \times (0) = 0$." + "\n" + r"2. Force in non-uniform field: $\vec{F} = (\vec{p} \cdot \nabla)\vec{E} = p_0 \frac{\partial}{\partial x}(E_0 x \hat{i} + E_0 y \hat{j}) = p_0 E_0 \hat{i}$.",
            "visual_tags": ["dipole_vector", "electric_field_grid"]
        }
    ],
    "Chemistry": [
        {
            "id": "CHEM-2023-ADV-P1-03",
            "exam": "JEE Advanced 2023",
            "topic": "VSEPR Theory & Trigonal Bipyramidal Geometry",
            "difficulty": "Medium",
            "question": r"Explain the difference in bond lengths and reactivities of axial vs equatorial bonds in $PCl_5$ gas using VSEPR theory and hybridization.",
            "solution": r"1. Hybridization: $sp^3d$ utilizing $s, p_x, p_y, p_z, d_{z^2}$ orbitals." + "\n" + r"2. Three equatorial bonds form $120^\circ$ angles (bond length $\approx 202\text{ pm}$)." + "\n" + r"3. Two axial bonds form $90^\circ$ angles with 3 equatorial bonds each, experiencing stronger repulsion and thus longer bonds ($\approx 240\text{ pm}$)." + "\n" + r"4. Axial bonds are weaker and cleaved first upon heating: $PCl_5(g) \xrightarrow{\Delta} PCl_3(g) + Cl_2(g)$.",
            "visual_tags": ["trigonal_bipyramidal", "axial_equatorial_spheres", "bonds"]
        },
        {
            "id": "CHEM-2022-ADV-P2-09",
            "exam": "JEE Advanced 2022",
            "topic": "Organic Stereochemistry & Walden Inversion",
            "difficulty": "Hard",
            "question": r"Describe the transition state geometry, hybridization change, and stereochemical outcome during the $S_N2$ substitution of $(2R)\text{-2-bromobutane}$ with methoxide ion.",
            "solution": r"1. Backside attack: Methoxide ion attacks from $180^\circ$ opposite to $C-Br$ bond." + "\n" + r"2. Transition State: Central carbon transitions from $sp^3$ to a planar $sp^2$-like arrangement with partial bonds to $OCH_3$ and $Br$." + "\n" + r"3. Outcome: $100\%$ stereochemical inversion (Walden inversion), producing $(2S)\text{-2-methoxybutane}$.",
            "visual_tags": ["walden_inversion", "transition_state_arrows", "tetrahedral_flip"]
        },
        {
            "id": "CHEM-2023-MAIN-APR-11",
            "exam": "JEE Main 2023",
            "topic": "Coordination Chemistry & Crystal Field Theory",
            "difficulty": "Medium",
            "question": r"Calculate the Crystal Field Stabilization Energy (CFSE) and magnetic moment for $[Fe(CN)_6]^{4-}$ vs $[Fe(H_2O)_6]^{2+}$.",
            "solution": r"1. $Fe^{2+}$ has $3d^6$ configuration." + "\n" + r"2. With strong field ligand $CN^-$: Low spin $t_{2g}^6 e_g^0$. CFSE $= -0.4\Delta_o \times 6 + 2P = -2.4\Delta_o + 2P$, Diamagnetic ($\mu = 0\text{ BM}$)." + "\n" + r"3. With weak field ligand $H_2O$: High spin $t_{2g}^4 e_g^2$. CFSE $= (-0.4\times 4 + 0.6\times 2)\Delta_o = -0.4\Delta_o$, Paramagnetic ($\mu = \sqrt{4(4+2)} = 4.90\text{ BM}$).",
            "visual_tags": ["octahedral_splitting", "electron_spin_arrows"]
        }
    ],
    "Mathematics": [
        {
            "id": "MATH-2023-ADV-P1-05",
            "exam": "JEE Advanced 2023",
            "topic": "3D Planes & Vector Normal Geometry",
            "difficulty": "Hard",
            "question": r"Find the vector equation of the plane passing through the line of intersection of planes $P_1: \vec{r}\cdot(\hat{i}+\hat{j}+\hat{k}) = 1$ and $P_2: \vec{r}\cdot(2\hat{i}+3\hat{j}-\hat{k}) = -4$, perpendicular to $P_3: \vec{r}\cdot(\hat{i}-\hat{j}+\hat{k}) = 0$.",
            "solution": r"1. Equation of family of planes: $\vec{r} \cdot [(\hat{i}+\hat{j}+\hat{k}) + \lambda(2\hat{i}+3\hat{j}-\hat{k})] = 1 - 4\lambda$." + "\n" + r"2. Normal vector $\vec{n} = (1+2\lambda)\hat{i} + (1+3\lambda)\hat{j} + (1-\lambda)\hat{k}$." + "\n" + r"3. Perpendicularity with $P_3$: $\vec{n} \cdot (\hat{i}-\hat{j}+\hat{k}) = 0 \implies (1+2\lambda) - (1+3\lambda) + (1-\lambda) = 0$." + "\n" + r"4. Yields $\lambda = 0.5$. Substituting gives the plane: $4x + 5y + z + 2 = 0$.",
            "visual_tags": ["plane_grid", "normal_vector_arrow", "intersection_line"]
        },
        {
            "id": "MATH-2022-ADV-P2-02",
            "exam": "JEE Advanced 2022",
            "topic": "Shortest Distance Between Skew Lines",
            "difficulty": "Hard",
            "question": r"Find the shortest distance between skew lines $L_1: \vec{r} = (\hat{i}+2\hat{j}+3\hat{k}) + t(2\hat{i}+3\hat{j}+4\hat{k})$ and $L_2: \vec{r} = (2\hat{i}+4\hat{j}+5\hat{k}) + s(3\hat{i}+4\hat{j}+5\hat{k})$.",
            "solution": r"1. $\vec{a}_1 = (1,2,3), \vec{b}_1 = (2,3,4)$, $\vec{a}_2 = (2,4,5), \vec{b}_2 = (3,4,5)$." + "\n" + r"2. $\vec{b}_1 \times \vec{b}_2 = -\hat{i} + 2\hat{j} - \hat{k}$. Magnitude $|\vec{b}_1 \times \vec{b}_2| = \sqrt{1+4+1} = \sqrt{6}$." + "\n" + r"3. $\vec{a}_2 - \vec{a}_1 = \hat{i} + 2\hat{j} + 2\hat{k}$." + "\n" + r"4. Shortest Distance: $d = \frac{|(\vec{a}_2 - \vec{a}_1) \cdot (\vec{b}_1 \times \vec{b}_2)|}{|\vec{b}_1 \times \vec{b}_2|} = \frac{|-1 + 4 - 2|}{\sqrt{6}} = \frac{1}{\sqrt{6}}$.",
            "visual_tags": ["skew_lines", "shortest_distance_segment", "cross_product_vector"]
        }
    ]
}

def save_and_sync_json(subject: str, new_items: List[Dict[str, Any]]):
    filename = SUBJECT_FILE_MAP.get(subject, "physics.json")
    file_path = DATA_DIR / filename
    
    existing_items: List[Dict[str, Any]] = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_items = json.load(f)
        except Exception:
            existing_items = []
            
    existing_ids = {item.get("id") for item in existing_items if "id" in item}
    
    added_count = 0
    for item in new_items:
        if item.get("id") not in existing_ids:
            existing_items.append(item)
            existing_ids.add(item.get("id"))
            added_count += 1
            
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_items, f, ensure_ascii=False, indent=2)
        
    print(f"✅ [{subject}] Saved {added_count} items -> {filename} (Total: {len(existing_items)})")

# =============================================================================
# 2. PDF Chunking & Gemini 구조화 파싱
# =============================================================================

PDF_PARSE_PROMPT = """You are an expert IIT-JEE Question Digitizer.
Extract all exam questions from this text chunk into a JSON array:
[
  {
    "id": "Short slug e.g. PHY-ADV-2023-P1-Q03",
    "exam": "JEE Advanced/Main + Year",
    "subject": "Physics" | "Chemistry" | "Mathematics",
    "topic": "Chapter/Concept title",
    "difficulty": "Easy" | "Medium" | "Hard",
    "question": "Full question with complete LaTeX math ($...$ for inline, $$...$$ for display)",
    "solution": "Step-by-step mathematical derivation and final answer in LaTeX",
    "visual_tags": ["relevant", "3d", "tags"]
  }
]
Output ONLY valid JSON without markdown fences."""

def process_pdf_files():
    pdf_files = glob.glob(str(RAW_DIR / "*.pdf"))
    if not pdf_files:
        print(f"ℹ️ No PDF files found in '{RAW_DIR}'. Place exam PDFs there.")
        return

    print(f"📂 Found {len(pdf_files)} PDF file(s) in {RAW_DIR}...")
    
    for pdf_path in pdf_files:
        file_name = Path(pdf_path).name
        print(f"\n📄 Reading PDF: {file_name}...")
        
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"   Total Pages: {total_pages}")
            
            # 3페이지씩 묶어서 배치 처리
            batch_size = 3
            for start_idx in range(0, total_pages, batch_size):
                end_idx = min(start_idx + batch_size, total_pages)
                print(f"   -> Processing Pages {start_idx + 1} to {end_idx}...")
                
                chunk_text = ""
                for page_num in range(start_idx, end_idx):
                    page_content = reader.pages[page_num].extract_text() or ""
                    chunk_text += f"\n--- Page {page_num + 1} ---\n{page_content}"
                
                if len(chunk_text.strip()) < 50:
                    continue

                try:
                    response = ai_client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=f"{PDF_PARSE_PROMPT}\n\nExam PDF Text:\n{chunk_text}",
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            response_mime_type="application/json"
                        )
                    )

                    parsed_list = json.loads(response.text or "[]")
                    if not isinstance(parsed_list, list):
                        parsed_list = [parsed_list]

                    # 과목별 분류 저장
                    by_subject: Dict[str, List[Dict[str, Any]]] = {"Physics": [], "Chemistry": [], "Mathematics": []}
                    for q in parsed_list:
                        sub = q.get("subject", "Physics")
                        if sub not in by_subject:
                            topic_str = (q.get("topic", "") + " " + q.get("question", "")).lower()
                            if any(k in topic_str for k in ["chemistry", "acid", "vsepr", "bond", "orbital", "reaction", "carbon", "fe("]):
                                sub = "Chemistry"
                            elif any(k in topic_str for k in ["plane", "integral", "derivative", "matrix", "vector", "probability", "skew"]):
                                sub = "Mathematics"
                            else:
                                sub = "Physics"
                        by_subject[sub].append(q)

                    for sub, items in by_subject.items():
                        if items:
                            save_and_sync_json(sub, items)

                except Exception as inner_e:
                    print(f"   ⚠️ Batch error ({start_idx+1}-{end_idx}): {inner_e}")

        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")

# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ChatJEEPT Bulk Importer")
    parser.add_argument("--seed", action="store_true", help="Seed curated JEE questions")
    parser.add_argument("--pdf", action="store_true", help="Parse PDF files from raw_papers/")
    args = parser.parse_args()

    if not args.seed and not args.pdf:
        print("💡 Usage:")
        print("  python bulk_importer.py --seed   (Seed curated benchmark questions)")
        print("  python bulk_importer.py --pdf    (Extract and parse exam PDFs from raw_papers/)")
        return

    if args.seed:
        print("\n🚀 Seeding Curated Benchmark Dataset...")
        for subject, items in SEED_DATA.items():
            save_and_sync_json(subject, items)

    if args.pdf:
        print("\n🚀 Scanning and Parsing PDF Files...")
        process_pdf_files()

    print("\n🔄 Re-indexing ChromaDB Vector Collections...")
    init_and_populate_vector_db()
    print("\n🎉 All Done! ChromaDB is fully synchronized.")

if __name__ == "__main__":
    main()