'use client';

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import GraphCanvas from '../../components/GraphCanvas';
import NodeDetailsPanel from '../../components/NodeDetailsPanel';
import { graphAPI, GraphData, Node } from '../../lib/api';

export default function GraphPage() {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showPanel, setShowPanel] = useState(false);

  const { data: graphData, isLoading, error } = useQuery<GraphData>({
    queryKey: ['graph'],
    queryFn: () => graphAPI.getGraph(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const handleNodeClick = (node: Node) => {
    setSelectedNodeId(node.id);
  };

  const handleNodeDoubleClick = (node: Node) => {
    setSelectedNodeId(node.id);
    setShowPanel(true);
  };

  const handleClosePanel = () => {
    setShowPanel(false);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-white text-xl">Loading graph...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-red-400 text-xl">Error loading graph</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Main Graph Area */}
      <div className="flex-1 relative">
        <div className="absolute top-4 left-4 z-10 bg-gray-800 text-white p-4 rounded-lg shadow-lg">
          <h1 className="text-xl font-bold mb-2">Mind Map AI</h1>
          <div className="text-sm text-gray-400">
            <p>Nodes: {graphData?.nodes.length || 0}</p>
            <p>Edges: {graphData?.edges.length || 0}</p>
          </div>
        </div>

        <div className="absolute top-4 right-4 z-10 bg-gray-800 text-white p-2 rounded-lg shadow-lg">
          <div className="text-xs space-y-1">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              <span>Concept</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              <span>Person</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
              <span>Place</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
              <span>Idea</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
              <span>Event</span>
            </div>
          </div>
        </div>

        {graphData && (
          <GraphCanvas
            data={graphData}
            onNodeClick={handleNodeClick}
            onNodeDoubleClick={handleNodeDoubleClick}
            selectedNodeId={selectedNodeId || undefined}
          />
        )}
      </div>

      {/* Side Panel */}
      {showPanel && selectedNodeId && (
        <div className="border-l border-gray-700">
          <NodeDetailsPanel nodeId={selectedNodeId} onClose={handleClosePanel} />
        </div>
      )}
    </div>
  );
}
