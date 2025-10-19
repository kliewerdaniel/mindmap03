# User Guide - Mind Map AI

Welcome to Mind Map AI! This guide will help you get started with transforming your notes into intelligent knowledge graphs.

## Quick Start

### 1. Installation

**Prerequisites:**
- Python 3.8 or higher
- Node.js 18 or higher
- Ollama (recommended) or Llama.cpp for local LLM

**Setup Steps:**
```bash
# Clone the repository
git clone <repository-url>
cd mindmap-ai

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### 2. First Launch

1. **Start the backend:**
```bash
cd backend
uvicorn app.main:app --reload
```

2. **Start the frontend:**
```bash
cd frontend
npm run dev
```

3. **Open your browser** and navigate to `http://localhost:3000`

## Core Workflow

### Uploading Notes

1. **Click "Upload Notes"** in the main interface
2. **Drag and drop files** or click to browse:
   - Plain text (.txt)
   - Markdown (.md)
   - PDF documents (.pdf)
   - Word documents (.docx)
   - HTML files (.html)

3. **Configure processing options:**
   - Select LLM model (if multiple available)
   - Set confidence threshold (default: 0.7)
   - Choose entity types to extract

4. **Click "Process Notes"** to start extraction

### Exploring Your Knowledge Graph

#### Navigation Basics
- **Zoom**: Mouse wheel or pinch gestures
- **Pan**: Click and drag to move around
- **Node Selection**: Click on any node to see details
- **Relationship Focus**: Double-click relationships to highlight connections

#### Graph Elements
- **Blue Nodes**: People and individuals
- **Green Nodes**: Organizations and institutions
- **Purple Nodes**: Concepts and abstract ideas
- **Orange Nodes**: Locations and places
- **Red Nodes**: Events and occurrences

### Semantic Search

#### Text Search
1. **Enter natural language queries** in the search bar
2. **Examples:**
   - "machine learning algorithms"
   - "people working on climate change"
   - "companies in San Francisco"

#### Advanced Search
- **Filter by entity type**: Click type buttons to show only specific categories
- **Date range filtering**: Limit results to specific time periods
- **Confidence filtering**: Show only high-confidence extractions

### Graph Customization

#### Visual Settings
- **Layout Algorithm**: Choose from force-directed, hierarchical, or circular layouts
- **Node Spacing**: Adjust how closely packed nodes appear
- **Edge Styling**: Customize relationship line appearance
- **Color Themes**: Switch between light and dark modes

#### Filtering and Focus
- **Hide Node Types**: Temporarily remove certain entity categories
- **Show Only Connected**: Display only nodes with relationships
- **Expand/Collapse**: Focus on specific subgraphs

## Advanced Features

### Batch Processing

Process multiple notes simultaneously:
1. **Select multiple files** during upload
2. **Configure batch settings** for consistent processing
3. **Monitor progress** with the batch processing dashboard

### Export Options

Save your knowledge graphs in multiple formats:
- **GraphML**: For use in other graph analysis tools
- **JSON**: For programmatic access or backup
- **PNG/SVG**: High-quality images for presentations
- **PDF**: Publication-ready documents with metadata

### Keyboard Shortcuts

- `Ctrl/Cmd + K`: Focus search bar
- `Ctrl/Cmd + F`: Toggle fullscreen
- `Ctrl/Cmd + E`: Export current view
- `Ctrl/Cmd + S`: Save graph state
- `Delete/Backspace`: Remove selected elements
- `Escape`: Deselect all elements

## Configuration

### LLM Settings

Access through Settings → LLM Configuration:

- **Model Selection**: Choose from available local models
- **Temperature**: Control creativity vs. consistency (0.1-1.0)
- **Max Tokens**: Limit response length for large documents
- **Context Window**: Available working memory for processing

### Processing Parameters

Fine-tune extraction behavior:

- **Entity Threshold**: Minimum confidence for entity recognition
- **Relationship Threshold**: Minimum confidence for relationship extraction
- **Deduplication**: How aggressively to merge similar entities
- **Language Detection**: Automatically detect and process multiple languages

### Storage Settings

Configure where data is stored:

- **Graph Directory**: Location for NetworkX graph files
- **Embedding Directory**: Vector database storage location
- **Metadata Database**: SQLite database path
- **Backup Frequency**: Automatic backup scheduling

## Best Practices

### Note Preparation

For best results:
- **Use clear, structured writing**
- **Include proper names** (people, places, organizations)
- **Add context** around important concepts
- **Use consistent terminology**

### Processing Tips

- **Start small**: Process a few notes first to understand extraction quality
- **Iterate**: Use feedback to improve entity and relationship detection
- **Combine sources**: Merge graphs from related topics for richer connections
- **Regular maintenance**: Periodically review and clean up low-confidence extractions

### Performance Optimization

- **Hardware**: Use GPU acceleration when available for faster processing
- **Batch size**: Process notes in reasonable batches (10-50 files)
- **Memory management**: Monitor RAM usage for large graphs
- **Storage**: Keep sufficient disk space for graph expansion

## Troubleshooting

### Common Issues

**Slow Processing:**
- Reduce batch size
- Lower LLM temperature
- Use smaller context windows

**Poor Entity Recognition:**
- Check note formatting and clarity
- Adjust confidence thresholds
- Try different LLM models

**Graph Performance:**
- Enable graph simplification
- Use filtering to focus on important nodes
- Consider graph partitioning for very large datasets

**Import/Export Problems:**
- Verify file formats and encoding
- Check disk space and permissions
- Review logs for detailed error messages

### Getting Help

1. **Check the logs**: Look in `/backend/logs/` for detailed error information
2. **Review settings**: Ensure all paths and configurations are correct
3. **Test with sample data**: Use provided example files to verify functionality
4. **Community support**: Check GitHub issues and discussions

## Privacy and Security

### Data Protection

- **All processing is local**: No data leaves your machine
- **Encryption options**: Enable at-rest encryption for sensitive graphs
- **Access control**: Set up user authentication for shared systems
- **Backup security**: Encrypt backups containing sensitive information

### Privacy Features

- **No telemetry**: Zero tracking or analytics collection
- **Local models only**: No external API dependencies
- **Data sovereignty**: Complete control over your knowledge graph
- **GDPR compliance**: Designed for privacy regulation compliance

## What's Next?

### Roadmap Features

- **Multi-modal support**: Process PDFs, images, and audio
- **Advanced analytics**: Graph metrics and insights
- **Collaboration tools**: Share graphs with trusted partners
- **Mobile applications**: Access your knowledge on the go
- **Plugin ecosystem**: Extend functionality with community plugins

### Learning Resources

- **API Documentation**: Comprehensive guide for developers
- **Architecture Overview**: Deep dive into system design
- **Video Tutorials**: Step-by-step walkthroughs (coming soon)
- **Community Examples**: Real-world usage scenarios

---

**Happy knowledge mapping!** 🧠✨

*Remember: Your knowledge graph is as powerful as the connections you create. Start small, iterate often, and watch your understanding grow.*
