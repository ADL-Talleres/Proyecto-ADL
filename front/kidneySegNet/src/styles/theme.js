// theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#808080',
    },
    secondary: {
      main: '#ffffff',
    },
    background: {
      default: '#000000',
    },
    text: {
      main: '#ffffff',
      primary: '#ffffff',
      secondary: '#ffffff',
    },
    success: {
      main: '#93B7BE',
    },
    divider: '#D3D3D3',
  },
  typography: {
    fontFamily: 'monospace',
    h1: {
      color: '#FFFFFF',
    },
    h2: {
      color: '#FFFFFF',
    },
    h3: {
      color: '#FFFFFF',
    },
    h4: {
      color: '#FFFFFF',
    },
    h5: {
      color: '#FFFFFF',
    },
    h6: {
      color: '#FFFFFF',
    },
    body1: {
      color: '#FFFFFF',
    },
    body2: {
      color: '#FFFFFF',
    },
  },
});

export default theme;
