'use client';

import React, { useEffect, useState } from 'react';
import { Node, graphAPI } from '../lib/api';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface NodeDetailsPanelProps {
  nodeId: string;
  onClose: () => void;
}

const NodeDetailsPanel: React.FC<NodeDetailsPanelProps> = ({ nodeId, onClose }) => {
  const [node, setNode] = useState<Node | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNode = async () => {
      try {
        setLoading(true);
        const nodeData = await graphAPI.getNode(nodeId);
        setNode(nodeData);
        setError(null);
      } catch (err) {
        setError('Failed to load node details');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchNode();
  }, [nodeId]);

  if (loading) {
    return (
      <div className="w-96 bg-gray-800 text-white p-6 shadow-lg">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-700 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-700 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (error || !node) {
    return (
      <div className="w-96 bg-gray-800 text-white p-6 shadow-lg">
        <div className="flex justify-between items-start mb-4">
          <h2 className="text-xl font-bold text-red-400">Error</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>
        <p>{error || 'Node not found'}</p>
      </div>
    );
  }

  return (
    <div className="w-96 bg-gray-800 text-white p-6 shadow-lg overflow-y-auto max-h-screen">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-2xl font-bold">{node.label}</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white">
          <XMarkIcon className="w-6 h-6" />
        </button>
      </div>

      <div className="space-y-4">
        {/* Node Type */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase mb-1">Type</h3>
          <span className="inline-block px-3 py-1 bg-blue-600 rounded-full text-sm">
            {node.type}
          </span>
        </div>

        {/* Confidence */}
        {node.confidence && (
          <div>
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-1">Confidence</h3>
            <div className="flex items-center">
              <div className="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${node.confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-sm">{(node.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}

        {/* Provenance */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">
            Provenance ({node.provenance?.length || 0} sources)
          </h3>
          {node.provenance && node.provenance.length > 0 ? (
            <div className="space-y-2">
              {node.provenance.map((prov, idx) => (
                <div key={idx} className="bg-gray-700 p-3 rounded text-sm">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Note ID: {prov[0]}</span>
                    <span>Span: {prov[1]}-{prov[2]}</span>
                  </div>
                  <button
                    className="text-blue-400 hover:text-blue-300 text-xs"
                    onClick={() => {
                      // TODO: Navigate to note or show excerpt
                      console.log('View note:', prov[0]);
                    }}
                  >
                    View source →
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No provenance data available</p>
          )}
        </div>

        {/* Metadata */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Metadata</h3>
          <div className="bg-gray-700 p-3 rounded text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-400">ID:</span>
              <span className="font-mono">{node.id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Created:</span>
              <span>{new Date(node.created_at).toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Updated:</span>
              <span>{new Date(node.updated_at).toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="pt-4 border-t border-gray-700">
          <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded mb-2">
            Edit Node
          </button>
          <button className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded">
            Find Related
          </button>
        </div>
      </div>
    </div>
  );
};

export default NodeDetailsPanel;
