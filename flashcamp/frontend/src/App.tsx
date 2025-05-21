import React from "react";
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import NavBar from './components/NavBar';
import theme from './theme';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import WizardPage from './pages/WizardPage';
import ResultsPage from './pages/ResultsPage';
import HomePage from './pages/HomePage';
import { PillarScores } from './types/api';

const App: React.FC = () => {
  // Initialize with default values to satisfy TypeScript
  const demoResult = {
    // Individual scores
    capital_score: 0,
    advantage_score: 0,
    market_score: 0,
    people_score: 0,
    overall_score: 0,
    
    // Pillar scores (v2 model)
    pillar_scores: {
      capital: 0,
      advantage: 0,
      market: 0,
      people: 0
    } as PillarScores,
    
    // Core properties
    success_probability: 0,
    startup_id: "demo",
    startup_name: "Demo Startup",
    
    // Other required fields
    alerts: [] as { type: string, message: string, severity: string }[],
    insights: [],
    recommendations: []
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <NavBar />
        <Routes>
          <Route path="/analyze" element={<WizardPage />} />
          <Route path="/results" element={<ResultsPage result={demoResult} />} />
          <Route path="/" element={<HomePage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
};

export default App;
