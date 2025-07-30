# 🤝 Contributing to Smart Segments

Thank you for your interest in contributing to Smart Segments! This document provides guidelines and information for contributors.

## 🎯 Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Install** development dependencies: `./dev-install.sh`
4. **Create** a feature branch: `git checkout -b feature/amazing-feature`
5. **Make** your changes
6. **Test** your changes: `./dev-utils.sh test`
7. **Commit** your changes: `git commit -m "Add amazing feature"`
8. **Push** to your fork: `git push origin feature/amazing-feature`
9. **Create** a Pull Request

## 🛠️ Development Setup

### Prerequisites
- Krita 5.0+ with Python support
- Python 3.7+
- Git

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Smart_Segments.git
cd Smart_Segments

# Install development environment
./dev-install.sh

# Verify installation
./dev-utils.sh test
```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small

### Testing
- Write tests for new features
- Ensure all tests pass before submitting PR
- Include both unit and integration tests when applicable

### Documentation
- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update changelog.md with your changes

## 🐛 Bug Reports

When reporting bugs, please include:
- **Krita version** and operating system
- **Plugin version**
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Screenshots** or logs if applicable

## ✨ Feature Requests

For feature requests:
- Describe the feature and its use case
- Explain why it would be valuable
- Consider implementation complexity
- Provide mockups or examples if helpful

## 📝 Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Update changelog.md** with your changes
4. **Ensure CI passes** (tests, linting)
5. **Request review** from maintainers

### PR Title Format
- `feat: add new feature description`
- `fix: resolve specific issue`
- `docs: update documentation`
- `test: add or update tests`
- `refactor: improve code structure`

## 🏗️ Project Structure

```
📦 Smart Segments
├── 🎨 pykrita/smart_segments/     # Main plugin code
│   ├── core/                     # Core functionality
│   ├── ui/                       # User interface
│   └── utils/                    # Utility functions
├── 🧪 tests/                     # Test suite
├── 📚 documents/                 # Documentation
├── 🎁 resources/                 # Assets and resources
└── 🔧 dev-*.sh                   # Development scripts
```

## 🎯 Areas for Contribution

- **AI/ML**: Improve segmentation algorithms
- **UI/UX**: Enhance user interface and experience
- **Performance**: Optimize speed and memory usage
- **Testing**: Expand test coverage
- **Documentation**: Improve guides and examples
- **Platform Support**: Cross-platform compatibility
- **Accessibility**: Make the plugin more accessible

## 📞 Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: software+smartsegments@hawkins.tech for private matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<div align="center">

**Thank you for contributing to Smart Segments! 🎨**

*Together, we're making digital art creation more intelligent and intuitive.*

</div>
