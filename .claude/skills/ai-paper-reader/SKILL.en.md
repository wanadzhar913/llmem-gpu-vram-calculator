---
name: ai-paper-reader
description: Deeply analyze AI papers and generate publish-ready professional reading notes
---

# AI Paper Reading Notes Generator

## Core Goal

Generate reading notes for papers that are **ready to publish directly on technical communities** (Zhihu, Juejin, WeChat Official Accounts, etc.).

Note requirements:
- **Complete content**: No omission of core technical details; deep elaboration of innovations
- **Professional and readable**: Technical blog style — both deep and easy to understand
- **Objective and accurate**: Analysis based on the paper's content, no subjective speculation added
- **Deep thinking**: Help readers understand deeply through a Q&A section

---

## Writing Guidelines

### Do

1. **Professional, accurate wording**
   - Use terminology standard within the field
   - Formulas and symbols must strictly match the original paper
   - Technical details described clearly, without ambiguity

2. **Explain from simple to deep**
   - Give intuition first for complex concepts, then details
   - Use analogies to help understand abstract concepts
   - Explain the meaning of each variable in formulas, term by term

3. **Clearly structured organization**
   - Clear logical hierarchy
   - Highlight key content
   - Use diagrams/charts appropriately to aid explanation

4. **Valuable in-depth analysis**
   - Analyze the reasoning behind design choices
   - Compare similarities/differences with related work
   - Point out the method's applicable scope and limitations

### Must Avoid

1. **AI clichés and template phrasing**
   - ❌ "The core contribution of this paper is..."
   - ❌ "The advantage of this method lies in..."
   - ❌ "In summary..."
   - ❌ "It is worth noting that..."
   - ❌ "Has important significance / broad application prospects..."

2. **Empty summaries and evaluations**
   - ❌ "This is an important piece of work"
   - ❌ "Provides new ideas for this field"
   - ❌ Vague generalities without concrete analysis

3. **Excessive formatting decoration**
   - ❌ Heavy emoji use
   - ❌ Bolding every sentence
   - ❌ Too many nested heading levels

4. **Unnecessary first person**
   - ❌ "I think..."
   - ❌ "My understanding is..."
   - Maintain an objective narrative perspective

---

## Note Structure

### 0. Meta Information (start of the note)

Every note should begin with the following info block, so readers can quickly decide whether to keep reading:

```markdown
> **Paper**: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
> **Authors**: Meta AI
> **Published**: ICML 2024
> **Reading time**: ~15 minutes
> **Difficulty**: ⭐⭐⭐⭐ (requires Transformer, recommendation systems basics)
> **Prerequisites**: Attention mechanism, DLRM, Scaling Law concept
```

**Difficulty levels**:
- ⭐ Beginner: no specialized background needed
- ⭐⭐ Basic: familiarity with deep learning fundamentals
- ⭐⭐⭐ Intermediate: familiar with the relevant field
- ⭐⭐⭐⭐ Professional: requires solid domain knowledge
- ⭐⭐⭐⭐⭐ Expert: involves complex math or cutting-edge research

### 1. TL;DR

Summarize the paper's core innovation in 2–3 sentences, so readers without time for a full read can quickly grasp the key point.

```markdown
## TL;DR

Traditional recommendation models like DLRM rely on heavy manual feature engineering and can't scale. This paper reframes recommendation as a sequence generation problem. The core innovation is the HSTU architecture: replacing Softmax with Pointwise Attention to preserve the absolute strength of user preferences, allowing recommendation systems to exhibit an LLM-like Scaling Law for the first time.
```

**Requirements**:
- 2–3 sentences, no more than ~100 words
- Must include: problem background + core approach + key innovation
- Avoid vague generalities — must contain concrete technical points

### 2. Paper Overview

Concisely answer three questions:
- **What problem does it solve**: one-sentence description
- **Core approach**: one-sentence summary
- **Main contributions**: 2–3 bullet points

```markdown
## Paper Overview

**Problem**: Large-scale recommendation systems cannot continuously improve quality by increasing compute, unlike LLMs

**Approach**: Shift recommendation from "feature engineering + discriminative model" to "sequence modeling + generative model"

**Contributions**:
1. Proposes the Generative Recommenders (GRs) paradigm, achieving a Scaling Law for recommendation systems
2. Designs the HSTU architecture, replacing Softmax with Pointwise Attention to preserve intensity information
3. Proposes the M-FALCON inference algorithm for efficient candidate scoring
```

### 3. Background and Motivation

Explain the problems with existing methods and why a new method is needed:
- How existing methods work
- What problems/bottlenecks exist
- What the root cause of the problem is

### 4. Core Method (key section)

This is the core part of the note, requiring **completeness, depth, no omissions**.

#### Organization

1. **Overall architecture**
   - Provide an architecture diagram
   - Explain data flow
   - Annotate key modules

2. **Detailed explanation of each core module** (for each key module)
   - Input/output description
   - Core formulas + term-by-term explanation
   - Pseudocode/code implementation
   - Analysis of the reasoning behind design choices

3. **Key technical details**
   - Training strategy
   - Hyperparameter settings
   - Implementation tricks

#### Example format

```markdown
## Core Method

### Overall Architecture

[Architecture diagram]

Data flow: user history sequence → Embedding → HSTU Layers × L → prediction head

### HSTU Layer Details

#### Input/Output
- Input: X ∈ R^{N×d}, where N is sequence length, d is embedding dimension
- Output: Y ∈ R^{N×d}

#### Core Formulas

**Pointwise Projection**:
$$U, V, Q, K = \text{Split}(\phi_1(f_1(X)))$$

Where:
- $\phi_1$: SiLU activation function
- $f_1$: a single linear transformation layer
- Split divides the output into four vectors

**Spatial Aggregation**:
$$A(X)V(X) = \phi_2(Q(X)K(X)^T + r_{ab}) V(X)$$

Key point: uses SiLU instead of Softmax, preserving the absolute strength information of attention.

#### Code Implementation

```python
class HSTULayer(nn.Module):
    def forward(self, x):
        # Pointwise Projection
        projected = F.silu(self.proj_in(x))
        u, v, q, k = projected.split([...], dim=-1)

        # Spatial Aggregation (not Softmax!)
        attn = F.silu(q @ k.T + self.rel_bias)
        out = self.norm(attn @ v) * u

        return x + self.proj_out(out)
```

#### Design Analysis

**Why SiLU instead of Softmax?**

Softmax normalizes attention into a probability distribution, which loses important information in the recommendation setting...
```

### 5. Experimental Analysis

Not a list of numbers, but distilled key conclusions:

- **Main results**: key findings compared to baselines
- **Ablation studies**: contribution analysis of each component
- **Scaling analysis**: relationship between compute and performance (if present)
- **Limitations**: under what circumstances the method performs poorly

### 6. In-Depth Understanding Q&A

Use carefully designed questions to help readers deeply understand the paper's key points.

**Q&A shown directly, no collapsing/folding**.

```markdown
## In-Depth Understanding Q&A

### Q1: Why isn't Softmax Attention suitable for the recommendation setting?

Recommendation scenarios need to predict the **absolute strength** of user preference (e.g., watch duration), not just **relative ranking**.

Consider two users:
- User A: 10 historical interactions
- User B: 100 historical interactions

With Softmax, both users' attention weights get normalized to [0,1], causing the information "User B is more active" to be lost.

Pointwise Attention preserves the accumulated raw magnitude, so the model can learn the difference in activity level.

### Q2: How does HSTU replace the Transformer's 6 linear layers with just 2?

A standard Transformer layer needs:
- Q, K, V projections: 3 linear layers
- Output projection: 1 linear layer
- FFN: 2 linear layers (expand + compress)

HSTU's simplification:
1. **Fuse Q, K, V, U projections**: one linear layer generates all four vectors at once
2. **Replace FFN with U-gating**: `output * U` achieves a similar nonlinear transformation

The cost is reduced per-layer expressive power, but this is compensated by stacking more layers.

### Q3: Why does Stochastic Length training discard 70% of tokens with almost no performance loss?

The key lies in the **statistical properties** of user behavior:

1. **Temporal repetitiveness**: users repeatedly interact with similar content, so redundancy is high
2. **Low-rank interests**: 10,000 interactions may only involve about 20 main interest categories
3. **Recency priority**: sampling weights recent behavior more heavily, preserving the most relevant information

As long as the sample size exceeds a certain multiple of the number of interest categories, all interests can be covered with high probability.
```

### 7. Summary and Reflection

Objectively summarize the paper's contributions and limitations:

```markdown
## Summary

### Core Contributions
- Demonstrates that recommendation systems can follow a Scaling Law
- Proposes an Attention variant suited to the recommendation setting

### Limitations
- Cold-start scenarios: advantage is less clear when history sequences are too short
- Compute cost: requires substantial GPU resources
- Real-time latency: inference latency challenges for long sequences

### Applicable Scenarios
- Scenarios with rich user history (>100 interactions)
- Sufficient compute resources available
- Real-time requirements are not extremely strict
```

---

## Directory Structure Convention

Papers and reading notes should be organized under a unified subdirectory for easy management and retrieval:

```
paper-notes/
├── hstu/                           # one directory per paper, short name
│   ├── paper.pdf                   # original paper PDF
│   ├── README.md                   # reading notes (main file)
│   └── images/                     # extracted figures
│       ├── fig1_architecture.png
│       ├── fig2_method.png
│       └── fig3_scaling.png
│
├── attention-is-all-you-need/
│   ├── paper.pdf
│   ├── README.md
│   └── images/
│
└── din-deep-interest-network/
    ├── paper.pdf
    ├── README.md
    └── images/
```

**Naming conventions**:
- Directory name: paper's short name or keyword, lowercase, joined with `-`
- Notes file: always named `README.md`, for direct GitHub preview
- Image directory: always named `images/`

**Image naming convention**:
```
fig{number}_{type}_{brief description}.png

Types:
- arch: architecture diagram
- method: method flow diagram
- result: experimental results
- ablation: ablation study
- compare: comparison diagram

Examples:
- fig1_arch_overall.png
- fig2_method_attention.png
- fig3_result_scaling.png
```

---

## Q&A Section Design Guide

### Question Types

1. **Principle understanding**
   - Why is it designed this way?
   - What are the advantages compared to alternatives?

2. **Detail clarification**
   - The specific meaning of a symbol/operation
   - Distinguishing easily-confused concepts

3. **Boundary conditions**
   - Under what circumstances does the method fail?
   - What are the underlying assumptions?

4. **Extended thinking**
   - Can it be transferred to other scenarios?
   - What are possible directions for improvement?

### Answer Requirements

- **Show directly**: no folding/collapsing, readers can read smoothly
- **Well-reasoned**: answers need argumentation, not simple assertions
- **Use examples appropriately**: use concrete examples to aid understanding
- **Acknowledge uncertainty**: for parts the paper doesn't specify, it's fine to mark as "speculation"

---

## Figure/Chart Handling

### Figures That Must Be Extracted

- Overall architecture diagram
- Core method flow diagram
- Key experimental results (Scaling Law curves, etc.)

### Figure Caption Convention

```markdown
![Architecture diagram](images/fig1_architecture.png)

**Figure content**: HSTU's overall architecture, with DLRM comparison on the left

**Key information**:
- Input is a unified item-behavior alternating sequence
- HSTU Layers can be stacked indefinitely
- Output is a multi-task prediction head

**Corresponds to text**: described in detail in Section 3.2
```

---

## Figure Extraction Tools

Figures in academic papers come in two types, requiring different extraction methods:

| Type | Characteristics | Extraction method |
|------|-----------------|--------------------|
| **Embedded images** | PNG/JPEG inserted by the authors | `get_images()` |
| **Vector graphics** | Drawn figures like architecture/flow diagrams | `cluster_drawings()` |

### Method 1: Extract Embedded Images

Suitable for bitmaps directly inserted into the paper (e.g., screenshots of experimental results, photos, etc.):

```python
import fitz  # PyMuPDF
import os

def extract_embedded_images(pdf_path, output_dir):
    """Extract bitmap images embedded in the PDF"""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            image_ext = base["ext"]

            # Filter out images that are too small (likely icons/decorations)
            if base["width"] > 100 and base["height"] > 100:
                output_path = f"{output_dir}/page{page_num+1}_img{img_idx+1}.{image_ext}"
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

    doc.close()
```

### Method 2: Extract Vector Graphics (recommended)

Suitable for vector figures drawn in the paper, such as architecture diagrams, flowcharts, and charts:

```python
import fitz
import os

def extract_vector_figures(pdf_path, output_dir, dpi=200, min_size=100):
    """
    Use cluster_drawings() to identify vector-graphic regions and screenshot them

    Args:
        pdf_path: path to the PDF file
        output_dir: output directory
        dpi: output resolution (default 200; raise to 300 for clearer images)
        min_size: minimum size threshold to filter out decorative lines (default 100pt)
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    figures = []
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Identify clustered regions of vector graphics
        # x_tolerance/y_tolerance control the merge distance between adjacent elements
        try:
            drawing_rects = page.cluster_drawings(
                x_tolerance=3,
                y_tolerance=3
            )
        except Exception:
            # Some PDFs may not support this; skip
            continue

        for idx, rect in enumerate(drawing_rects):
            # Filter out regions that are too small (likely lines/decorations)
            if rect.width < min_size or rect.height < min_size:
                continue

            # Expand the bounding box to avoid cropping too tightly
            rect = rect + (-10, -10, 10, 10)
            # Ensure it doesn't exceed the page bounds
            rect = rect & page.rect

            # High-resolution screenshot
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=rect)

            output_path = f"{output_dir}/page{page_num+1}_fig{idx+1}.png"
            pix.save(output_path)
            figures.append({
                "page": page_num + 1,
                "path": output_path,
                "rect": rect
            })

    doc.close()
    return figures
```

### Method 3: Manually Specify a Region to Crop

When automatic detection doesn't work well, you can specify coordinates manually:

```python
import fitz

def crop_figure(pdf_path, page_num, rect, output_path, dpi=200):
    """
    Crop a specific region from a given page of a PDF

    Args:
        pdf_path: path to the PDF
        page_num: page number (starting from 1)
        rect: (x0, y0, x1, y1) coordinates in points (pt); 72pt = 1 inch
        output_path: output image path
        dpi: resolution
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    clip = fitz.Rect(rect)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, clip=clip)
    pix.save(output_path)
    doc.close()

# Usage example: crop a region on page 2
# Coordinates can be found via a PDF reader, or by first identifying with Method 2 and fine-tuning
crop_figure(
    "paper.pdf",
    page_num=2,
    rect=(50, 100, 550, 400),  # top-left (50,100) to bottom-right (550,400)
    output_path="./images/fig1_architecture.png"
)
```

### Smart Extraction (comprehensive approach)

Automatically tries multiple methods to extract all figures:

```python
import fitz
import os

def smart_extract_figures(pdf_path, output_dir, dpi=200):
    """
    Intelligently extract all figures/charts in a paper
    1. First use cluster_drawings to identify vector graphics
    2. Then extract embedded bitmaps
    3. Automatically filter and de-duplicate
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    results = {"vector": [], "embedded": []}

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 1. Extract vector graphics
        try:
            rects = page.cluster_drawings(x_tolerance=3, y_tolerance=3)
            for idx, rect in enumerate(rects):
                if rect.width > 100 and rect.height > 100:
                    rect = (rect + (-10, -10, 10, 10)) & page.rect
                    zoom = dpi / 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
                    path = f"{output_dir}/p{page_num+1}_vec{idx+1}.png"
                    pix.save(path)
                    results["vector"].append(path)
        except:
            pass

        # 2. Extract embedded images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            if base["width"] > 100 and base["height"] > 100:
                path = f"{output_dir}/p{page_num+1}_img{img_idx+1}.{base['ext']}"
                with open(path, "wb") as f:
                    f.write(base["image"])
                results["embedded"].append(path)

    doc.close()
    print(f"Extraction complete: {len(results['vector'])} vector figures, {len(results['embedded'])} bitmaps")
    return results

# Usage example
results = smart_extract_figures("paper.pdf", "./images/")
```

### FAQ

**Q: The extracted image contains multiple Figures merged together?**

Reduce the `x_tolerance` and `y_tolerance` parameters (e.g., 1–2) to make clustering stricter.

**Q: A single Figure got split into multiple pieces?**

Increase the tolerance parameters (e.g., 10–20) to merge adjacent elements.

**Q: Some Figures weren't detected?**

1. They may be embedded images — try the `get_images()` method
2. Use the manual region-specification method

**Q: The image is blurry?**

Increase the `dpi` parameter to 300 or higher.

---

## Usage

### Basic Usage

```
Please read this paper and generate a professional reading note suitable for publishing on a technical community.
```

### Specifying Focus Areas

```
Please read this paper, focusing on:
1. The differences between HSTU and a standard Transformer
2. The setup and conclusions of the Scaling Law experiments
3. Feasibility for deployment in industrial settings
```

### Comparative Analysis

```
Please compare and analyze how these two papers solve problem XXX differently.
```

---

## Technical Requirements

### Content Completeness
- Core formulas must be included, explained term by term
- Key algorithms must have pseudocode implementations
- Important hyperparameters and training details must not be omitted
- Key conclusions from ablation studies must be distilled

### Depth Requirements
- Analyze "why it's designed this way"
- Build connections with related work
- Point out the boundaries and limitations of the method

### Readability
- Intuition before details
- Pair code with formulas
- Break long formulas into step-by-step explanations

---

## Dependency Setup

```bash
# Image extraction
pip install pymupdf

# PDF to image (optional)
pip install pdf2image
```

### MCP Configuration (optional)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/papers"]
    },
    "notion": {
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {
        "OPENAPI_MCP_HEADERS": "{\"Authorization\": \"Bearer YOUR_TOKEN\", \"Notion-Version\": \"2022-06-28\"}"
      }
    }
  }
}
```
