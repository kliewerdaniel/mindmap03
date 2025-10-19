# Frontend Design System

## Visual Design Principles

### Color Palette

**Node Colors (by type):**
- Concept: `#3b82f6` (Blue)
- Person: `#10b981` (Green)
- Place: `#f59e0b` (Amber)
- Idea: `#8b5cf6` (Purple)
- Event: `#ef4444` (Red)
- Passage: `#6b7280` (Gray)

**UI Colors:**
- Background: `#111827` (Gray-900)
- Panel: `#1f2937` (Gray-800)
- Accent: `#fbbf24` (Yellow-400)
- Text Primary: `#ffffff`
- Text Secondary: `#9ca3af` (Gray-400)

### Visualization Cues

**Node Size:**
- Based on provenance count (number of source references)
- Formula: `min(20 + provenance_count * 5, 60)` pixels
- Larger nodes indicate concepts mentioned across multiple notes

**Edge Thickness:**
- Based on confidence weight (0-1)
- Formula: `1 + weight * 3` pixels
- Thicker edges indicate stronger relationships

**Node Selection:**
- Selected nodes have yellow (`#fbbf24`) border, 3px width
- Click to select, double-click to open details panel

### Layout Algorithm

**Graph Layout: COSE (Compound Spring Embedder)**
- Organic, force-directed layout
- Parameters:
  - Node repulsion: 8000
  - Ideal edge length: 100
  - Edge elasticity: 100
  - Animation duration: 500ms

### Interactions

**Primary Interactions:**
1. **Single Click Node**: Select node, highlight in graph
2. **Double Click Node**: Open NodeDetailsPanel with provenance
3. **Pan**: Click and drag on background
4. **Zoom**: Mouse wheel or pinch gesture
5. **Hover Node**: Show tooltip with label and type

**NodeDetailsPanel:**
- Slides in from right side
- Shows: Type, confidence, provenance list, metadata
- Actions: Edit node, find related nodes, view source notes

### Responsive Design

**Breakpoints:**
- Desktop: > 1024px (full graph + side panel)
- Tablet: 768-1024px (graph only, panel as overlay)
- Mobile: < 768px (not prioritized in Phase 5)

### Accessibility

- Keyboard navigation: Tab through nodes
- ARIA labels on interactive elements
- Sufficient color contrast (WCAG AA)
- Screen reader support for node metadata

## Component Structure

```
GraphPage
├── GraphCanvas (Cytoscape visualization)
│   ├── Node rendering
│   ├── Edge rendering
│   └── Event handlers
└── NodeDetailsPanel (Side panel)
    ├── Node metadata
    ├── Provenance list
    └── Action buttons
