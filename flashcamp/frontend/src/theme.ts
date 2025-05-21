import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: { main: '#312e81' }, // Deep blue
    secondary: { main: '#fbbf24' }, // Gold accent
    error: { main: '#ef4444' },
    background: { default: '#f8fafc' },
  },
  typography: {
    fontFamily: 'Inter, Roboto, Arial, sans-serif',
    h1: { fontWeight: 700 },
    h2: { fontWeight: 600 },
    h3: { fontWeight: 600 },
  },
});

export default theme;
