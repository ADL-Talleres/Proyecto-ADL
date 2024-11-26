import React from 'react';
import MainPageBlockerComponent from './MainPageBlockerComponent';
import HistoricComponent from './HistoricComponent';

const PrivateRoute = ({ children }) => {
  const isSigned = localStorage.getItem('signedIn') === 'true'; // Check the localStorage value

  return isSigned ? <HistoricComponent /> : <MainPageBlockerComponent />;
};

export default PrivateRoute;