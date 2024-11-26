import * as React from 'react';
import { useState, useEffect } from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import AdbIcon from '@mui/icons-material/Adb';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import { ThemeProvider } from '@mui/material/styles';
import { Link, useNavigate } from 'react-router-dom';
import theme from '../styles/theme';

function ResponsiveAppBar() {
  const [anchorEl, setAnchorEl] = useState(null);
  const [signedIn, setSignedIn] = useState(false);
  const [username, setUsername] = useState('');

  const navigate = useNavigate();

  useEffect(() => {
    const isSignedIn = localStorage.getItem('signedIn') === 'true';
    const storedUsername = localStorage.getItem('username');
    setSignedIn(isSignedIn);
    setUsername(storedUsername || '');
  }, []);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    localStorage.setItem('token', '');
    localStorage.setItem('username', '');
    localStorage.setItem('signedIn', 'false');
    setSignedIn(false);
    setUsername('');
    handleMenuClose();
    window.location.href = "/";
  };

  const renderButtons = () => {
    if (signedIn) {
      return (
        <>
          <Button
            key="newPrediction"
            sx={{ fontFamily: 'monospace', fontSize: '1.1rem', fontWeight: 500, mx: 1, color: 'text.primary' }}
            component={Link}
            to="/predict"
          >
            Make Prediction
          </Button>
          <Button
            key="username"
            sx={{ fontFamily: 'monospace', fontSize: '1.1rem', fontWeight: 500, mx: 1, color: 'text.primary' }}
            onClick={handleMenuOpen}
          >
            {username}
          </Button>
        </>
      );
    }
    return (
      <>
        <Button
          key="login"
          sx={{ fontFamily: 'monospace', fontSize: '1.1rem', fontWeight: 500, mx: 1, color: 'text.primary' }}
          component={Link}
          to="/login"
        >
          Login
        </Button>
      </>
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <AppBar position="static" sx={{ backgroundColor: 'transparent', boxShadow: 'none' }}>
        <Container maxWidth="xl" sx={{ py: 2 }}>
          <Toolbar
            disableGutters
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexDirection: { xs: 'column', md: 'row' },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <LocalHospitalIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1, color: 'text.primary' }} />
              <Typography
                variant="h6"
                noWrap
                component={Link}
                to="/"
                sx={{
                  mr: 2,
                  fontFamily: 'monospace',
                  fontWeight: 1000,
                  fontSize: '1.5rem',
                  letterSpacing: '.3rem',
                  color: 'text.primary',
                  textDecoration: 'none',
                  textAlign: { xs: 'center', md: 'left' },
                }}
              >
                KidneySegNet
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', flexDirection: { xs: 'column', md: 'row' } }}>
              {renderButtons()}
            </Box>
          </Toolbar>
        </Container>
      </AppBar>

      <Menu
        id="menu-appbar"
        anchorEl={anchorEl}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'center',
        }}
        keepMounted
        transformOrigin={{
          vertical: 'top',
          horizontal: 'center',
        }}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        sx={{
          '& .MuiPaper-root': {
            backgroundColor: 'secondary.main',
            color: 'text.main',
            mt: 1, // margin-top to ensure dropdown is below the username
          },
        }}
      >
        <MenuItem onClick={handleLogout}>Logout</MenuItem>
      </Menu>
    </ThemeProvider>
  );
}

export default ResponsiveAppBar;
