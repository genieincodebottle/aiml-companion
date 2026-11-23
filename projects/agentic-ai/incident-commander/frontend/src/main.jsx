import { StrictMode } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { ThemeModeProvider } from './ThemeModeProvider';

// The palettes themselves live in theme.js; ThemeModeProvider owns which one is
// active, persists the choice, and supplies CssBaseline.
ReactDOM.createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeModeProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeModeProvider>
  </StrictMode>
);
