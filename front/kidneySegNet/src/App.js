import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LoginComponent from './components/LoginComponent';
import SignupComponent from './components/SignupComponent';
import PredictionComponent from './components/PredictionComponent';
import ResponsiveAppBar from './components/HeaderComponent';
import PrivateRoute from './components/PrivateComponent';

const App = () => {
  return (
    <BrowserRouter>
      <ResponsiveAppBar />
      <Routes>
        <Route path="/login" element={<LoginComponent />} />
        <Route path="/signup" element={<SignupComponent />} />
        <Route path="/predict" element={<PredictionComponent />} />
        <Route path="/" element={<PrivateRoute />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
