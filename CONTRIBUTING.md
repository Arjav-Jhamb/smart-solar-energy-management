# Contributing to Smart Solar Energy Management System

First off, thank you for considering contributing to this project! 🎉

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by respect and professionalism. Please be kind and considerate.

### Our Standards
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

## How Can I Contribute?

### Reporting Bugs 🐛

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior**
- **Actual behavior**
- **Screenshots** (if applicable)
- **System information** (OS, Python version, Node version)

### Suggesting Enhancements 💡

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** - why is this enhancement useful?
- **Possible implementation** (if you have ideas)
- **Examples** from other projects (if applicable)

### Your First Code Contribution 🚀

Unsure where to begin? Look for issues labeled:
- `good first issue` - simple issues for beginners
- `help wanted` - issues that need attention

## Development Setup

### Prerequisites
- Python ≥ 3.8
- Node.js ≥ 16
- Git

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Running Tests
```bash
# Backend tests (when implemented)
cd backend
pytest

# Frontend tests (when implemented)
cd frontend
npm test
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/YourAmazingFeature
   ```

2. **Make your changes** following the style guidelines below

3. **Test your changes** thoroughly
   - Ensure backend runs without errors
   - Ensure frontend displays correctly
   - Test API endpoints
   - Check for console errors

4. **Commit your changes** with clear commit messages
   ```bash
   git commit -m "feat: add solar panel efficiency calculator"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/YourAmazingFeature
   ```

6. **Open a Pull Request** with:
   - Clear title describing the change
   - Description of what you changed and why
   - Screenshots (if UI changes)
   - Link to related issue (if applicable)

### PR Review Process
- At least one maintainer review required
- All tests must pass
- Code must follow style guidelines
- Documentation must be updated if needed

## Style Guidelines

### Python Code Style
- Follow **PEP 8** style guide
- Use **type hints** where applicable
- Maximum line length: **88 characters** (Black formatter)
- Use **meaningful variable names**

```python
# Good
def calculate_solar_efficiency(panel_output: float, panel_capacity: float) -> float:
    """Calculate solar panel efficiency percentage."""
    return (panel_output / panel_capacity) * 100

# Bad
def calc(x, y):
    return (x/y)*100
```

### TypeScript/React Code Style
- Use **functional components** with hooks
- Use **TypeScript** for type safety
- Follow **ESLint** rules
- Use **camelCase** for variables, **PascalCase** for components

```typescript
// Good
interface SolarData {
  timestamp: string;
  generation: number;
}

const SolarPanel: React.FC<{ data: SolarData }> = ({ data }) => {
  return <div>{data.generation} kW</div>;
};

// Bad
const solar_panel = (props) => {
  return <div>{props.data.generation}</div>;
};
```

### File Organization
```
backend/
  src/
    models/        # ML models
    api/           # API routes
    utils/         # Helper functions
    tests/         # Test files

frontend/
  src/
    components/    # React components
    hooks/         # Custom hooks
    utils/         # Helper functions
    types/         # TypeScript types
```

## Commit Message Guidelines

We follow the **Conventional Commits** specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```bash
feat(ml): add LSTM model for time-series prediction

fix(api): resolve WebSocket connection timeout issue

docs(readme): update installation instructions for Windows

refactor(dashboard): simplify data fetching logic

perf(backend): optimize CSV data loading
```

## Branch Naming

Use descriptive branch names:
- `feature/solar-efficiency-calculator`
- `bugfix/websocket-connection`
- `docs/api-documentation`
- `refactor/ml-pipeline`

## Testing Guidelines

### Backend Tests
```python
def test_solar_prediction():
    """Test solar power prediction accuracy."""
    features = {...}
    result = ml_manager.predict_solar(features)
    assert result > 0
    assert isinstance(result, float)
```

### Frontend Tests
```typescript
test('renders solar generation value', () => {
  render(<Dashboard currentData={mockData} />);
  expect(screen.getByText(/5.23 kW/i)).toBeInTheDocument();
});
```

## Documentation

- Update README.md for new features
- Add docstrings to Python functions
- Add JSDoc comments to TypeScript functions
- Update API documentation for new endpoints

## Questions?

Feel free to:
- Open an issue with the `question` label
- Email: arjavjhamb22@gmail.com

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- Project documentation

---

Thank you for contributing! 🙌

**Happy Coding!** ☀️💻