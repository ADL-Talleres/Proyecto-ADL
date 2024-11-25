import React from 'react';
import Box from '@mui/joy/Box';
import Container from '@mui/joy/Container';
import { Grid, CssBaseline, Typography, Button, Link } from '@mui/material';
import theme from '../styles/theme'; // Asegúrate de importar el mismo tema
import { ThemeProvider } from '@mui/system';

export default function MainPageBlockerComponent({ children, reversed }) {
  // Establecer overflow hidden en el body y el html para evitar desplazamiento

  return (
    <ThemeProvider theme={theme}>
      <Grid container component="main" sx={{ height: '100vh' }}>
        <CssBaseline />
        <Container
          sx={(theme2) => ({
            position: 'relative',
            minHeight: '100vh',
            display: 'flex',
            flexDirection: reversed ? 'column-reverse' : 'column',
            alignItems: 'center',
            py: 10,
            gap: 4,
            [theme2.breakpoints.up(834)]: {
              flexDirection: 'row',
              gap: 6,
            },
            [theme2.breakpoints.up(1199)]: {
              gap: 12,
            },
          })}
        >
          <Box
            sx={(theme2) => ({
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1rem',
              maxWidth: '50ch',
              textAlign: 'center',
              flexShrink: 999,
              [theme2.breakpoints.up(834)]: {
                minWidth: 420,
                alignItems: 'flex-start',
                textAlign: 'initial',
              },
            })}
          >
            <Typography variant="body1" sx={{ fontSize: '2rem', fontStyle: 'italic' }}>
            KidneySegNet leverages advanced U-Net architectures for precise segmentation of kidneys and renal tumors in CT scans, revolutionizing diagnostic efficiency and empowering early detection in clinical practice.
            </Typography>
            <Button variant="outlined" color="text" href="/signup" sx={{ mt: 2 , color: 'text.primary'}}>
              Get Started
            </Button>
            <Link href="/login" sx={{ mt: 1, color: 'text.primary' }}>
              Already a member? Sign in
            </Link>
          </Box>
          <Box
            sx={{
              minWidth: 300,
              maxWidth: '50%',
              borderRadius: 'sm',
              bgcolor: theme.palette.background.level2,
              flexBasis: '50%',
              textAlign: 'center',
            }}
          >
            <img
              src="https://www.uhhospitals.org/-/media/images/blog/2019/08/kidneys_blog-mainarticleimage.jpg?h=450&w=720&la=en&hash=560280915DF722B85916D420BAA8D117"
              alt=""
              style={{
                width: '100%',
                height: 'auto',
                maxHeight: '500px',
                borderRadius: 'sm',
              }}
            />
          </Box>
        </Container>
      </Grid>
    </ThemeProvider>
  );
}
