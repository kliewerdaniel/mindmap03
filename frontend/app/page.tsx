'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { graphAPI } from '../lib/api';
import NoteUploader from '../components/NoteUploader';
import Link from 'next/link';

export default function HomePage() {
  const { data: stats } = useQuery({
    queryKey: ['graph-stats'],
    queryFn: () => graphAPI.getStats(),
  });

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Mind Map AI</h1>
          <p className="text-sm text-gray-600 mt-1">
            Your personal knowledge graph, powered by local LLM
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500 uppercase">Nodes</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">
              {stats?.num_nodes || 0}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500 uppercase">Edges</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">
              {stats?.num_edges || 0}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500 uppercase">Density</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">
              {stats?.density?.toFixed(3) || '0.000'}
            </p>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white p-8 rounded-lg shadow mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Upload Notes
          </h2>
          <NoteUploader />
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link
            href="/graph"
            className="block p-6 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition"
          >
            <h3 className="text-xl font-bold mb-2">Explore Graph</h3>
            <p className="text-blue-100">
              Visualize and interact with your knowledge graph
            </p>
          </Link>

          <Link
            href="/search"
            className="block p-6 bg-purple-600 text-white rounded-lg shadow hover:bg-purple-700 transition"
          >
            <h3 className="text-xl font-bold mb-2">Semantic Search</h3>
            <p className="text-purple-100">
              Find related concepts and notes
            </p>
          </Link>
        </div>
      </main>
    </div>
  );
}
