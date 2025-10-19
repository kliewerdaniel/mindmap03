'use client';

import React, { useEffect, useRef, useState } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import Cytoscape from 'cytoscape';
import { GraphData, Node } from '../lib/api';

// We need to use any for Cytoscape styles due to complex type issues
// eslint-disable-next-line @typescript-eslint/no-explicit-any

interface GraphCanvasProps {
  data: GraphData;
  onNodeClick?: (node: Node) => void;
  onNodeDoubleClick?: (node: Node) => void;
  selectedNodeId?: string;
}

const GraphCanvas: React.FC<GraphCanvasProps> = ({
  data,
  onNodeClick,
  onNodeDoubleClick,
  selectedNodeId,
}) => {
  const cyRef = useRef<Cytoscape.Core | null>(null);
  const [elements, setElements] = useState<Cytoscape.ElementDefinition[]>([]);

  useEffect(() => {
    // Convert GraphData to Cytoscape elements
    const nodes = data.nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.label,
        type: node.type,
        confidence: node.confidence || 1,
        provenanceCount: node.provenance?.length || 0,
      },
    }));

    const edges = data.edges.map((edge, idx) => ({
      data: {
        id: `edge-${idx}`,
        source: edge.source,
        target: edge.target,
        label: edge.type,
        weight: edge.weight,
      },
    }));

    setElements([...nodes, ...edges]);
  }, [data]);

  useEffect(() => {
    if (cyRef.current && selectedNodeId) {
      // Highlight selected node
      cyRef.current.nodes().removeClass('selected');
      cyRef.current.getElementById(selectedNodeId).addClass('selected');
    }
  }, [selectedNodeId]);

  const stylesheet: any[] = [
    {
      selector: 'node',
      style: {
        'background-color': (ele: any) => {
          const type = ele.data('type');
          const colors: Record<string, string> = {
            concept: '#3b82f6',
            person: '#10b981',
            place: '#f59e0b',
            idea: '#8b5cf6',
            event: '#ef4444',
            passage: '#6b7280',
          };
          return colors[type] || '#9ca3af';
        },
        'label': 'data(label)',
        'width': (ele: any) => {
          const provCount = ele.data('provenanceCount') || 1;
          return Math.min(20 + provCount * 5, 60);
        },
        'height': (ele: any) => {
          const provCount = ele.data('provenanceCount') || 1;
          return Math.min(20 + provCount * 5, 60);
        },
        'font-size': '12px',
        'color': '#fff',
        'text-valign': 'center',
        'text-halign': 'center',
        'text-wrap': 'wrap',
        'text-max-width': '80px',
      },
    },
    {
      selector: 'node.selected',
      style: {
        'border-width': 3,
        'border-color': '#fbbf24',
      },
    },
    {
      selector: 'edge',
      style: {
        'width': (ele: any) => {
          const weight = ele.data('weight') || 0.5;
          return 1 + weight * 3;
        },
        'line-color': '#cbd5e1',
        'target-arrow-color': '#cbd5e1',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'label': 'data(label)',
        'font-size': '10px',
        'text-rotation': 'autorotate',
        'text-margin-y': -10,
      },
    },
  ];

  const layout = {
    name: 'cose',
    animate: true,
    animationDuration: 500,
    fit: true,
    padding: 30,
    nodeRepulsion: 8000,
    idealEdgeLength: 100,
    edgeElasticity: 100,
    nestingFactor: 1.2,
  };

  const handleCyReady = (cy: Cytoscape.Core) => {
    cyRef.current = cy;

    // Node click handler
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const nodeData = data.nodes.find((n) => n.id === node.id());
      if (nodeData && onNodeClick) {
        onNodeClick(nodeData);
      }
    });

    // Node double-click handler
    cy.on('dbltap', 'node', (evt) => {
      const node = evt.target;
      const nodeData = data.nodes.find((n) => n.id === node.id());
      if (nodeData && onNodeDoubleClick) {
        onNodeDoubleClick(nodeData);
      }
    });
  };

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      {elements.length > 0 ? (
        <CytoscapeComponent
          elements={elements}
          stylesheet={stylesheet}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
          cy={handleCyReady}
          zoom={1}
          pan={{ x: 0, y: 0 }}
          minZoom={0.3}
          maxZoom={3}
          wheelSensitivity={0.2}
        />
      ) : (
        <div className="flex items-center justify-center h-full text-gray-400">
          No graph data available. Ingest some notes to get started.
        </div>
      )}
    </div>
  );
};

export default GraphCanvas;
