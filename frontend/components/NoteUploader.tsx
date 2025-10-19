'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, DocumentTextIcon } from '@heroicons/react/24/outline';
import { ingestAPI } from '../lib/api';
import { useMutation, useQueryClient } from '@tanstack/react-query';

const NoteUploader: React.FC = () => {
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: (file: File) => ingestAPI.ingestFile(file),
    onSuccess: (data) => {
      setUploadStatus(`Successfully uploaded ${data.note_ids.length} note(s)`);
      // Invalidate graph query to trigger refresh
      queryClient.invalidateQueries({ queryKey: ['graph'] });
    },
    onError: (error) => {
      setUploadStatus(`Upload failed: ${error}`);
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadStatus(`Uploading ${file.name}...`);
      uploadMutation.mutate(file);
    }
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/markdown': ['.md'],
      'text/plain': ['.txt'],
      'application/zip': ['.zip'],
    },
    multiple: false,
  });

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />

        <CloudArrowUpIcon className="w-16 h-16 mx-auto mb-4 text-gray-400" />

        {isDragActive ? (
          <p className="text-lg text-blue-600">Drop the file here...</p>
        ) : (
          <div>
            <p className="text-lg text-gray-700 mb-2">
              Drag & drop a markdown file or zip archive here
            </p>
            <p className="text-sm text-gray-500">
              or click to select file
            </p>
            <p className="text-xs text-gray-400 mt-4">
              Supported: .md, .txt, .zip
            </p>
          </div>
        )}
      </div>

      {uploadStatus && (
        <div className="mt-4 p-4 bg-gray-100 rounded-lg">
          <p className="text-sm text-gray-700">{uploadStatus}</p>
        </div>
      )}

      {uploadMutation.isPending && (
        <div className="mt-4">
          <div className="animate-pulse flex items-center">
            <DocumentTextIcon className="w-5 h-5 mr-2 text-blue-500" />
            <span className="text-sm text-gray-600">Processing...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default NoteUploader;
