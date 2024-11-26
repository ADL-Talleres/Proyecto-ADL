// SignInSide.js
import React, { useState } from "react";
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import Typography from "@mui/material/Typography";
import { ThemeProvider } from "@mui/material/styles";
import signinimg from "../assets/image_login.png";
import theme from "../styles/theme";

function Copyright(props) {
    return (
        <Typography
            variant="body2"
            color="text.secondary"
            align="center"
            {...props}
        >
            {"Copyright © "}
            <Link color="inherit" href="https://mui.com/">
                Your Website
            </Link>{" "}
            {new Date().getFullYear()}
            {"."}
        </Typography>
    );
}

export default function SignInSide() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleSubmit = (event) => {
        event.preventDefault();
        if (email && password) {
            login();
        } else {
            alert("Please fill in both email and password");
        }
    };

    async function login() {
        const username = email.split("@")[0];

        try {
            const response = await fetch(
                `https://back-zu3yqmmklq-uc.a.run.app/users/login`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        email: email,
                        username: username,
                        password: password,
                    }),
                }
            );

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();
            localStorage.setItem("token", data.access_token);
            localStorage.setItem("signedIn", true);
            localStorage.setItem("username", data.username);
            localStorage.setItem("email", data.email);
            window.location.href = "/";
        } catch (error) {
            console.error(
                "There has been a problem with your fetch operation:",
                error
            );
            alert("Login failed. Please check your credentials and try again.");
        }
    }

    return (
        <ThemeProvider theme={theme}>
            <Grid container component="main" sx={{ height: "100vh" }}>
                <CssBaseline />
                <Grid
                    item
                    xs={false}
                    sm={4}
                    md={7}
                    sx={{
                        backgroundImage: `url(${signinimg})`,
                        backgroundColor: "#87CEEB",
                        backgroundSize: "cover",
                        backgroundPosition: "center",
                    }}
                />
                <Grid
                    item
                    xs={12}
                    sm={8}
                    md={5}
                    component={Paper}
                    elevation={6}
                    square
                    sx={{ backgroundColor: "background.default" }}
                >
                    <Box
                        sx={{
                            my: 8,
                            mx: 4,
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                        }}
                    >
                        <Avatar sx={{ m: 1, bgcolor: "text.secondary" }}>
                            <LockOutlinedIcon />
                        </Avatar>
                        <Typography component="h1" variant="h5">
                            Sign in
                        </Typography>
                        <Box
                            component="form"
                            noValidate
                            onSubmit={handleSubmit}
                            sx={{ mt: 1 }}
                        >
                            <TextField
                                margin="normal"
                                required
                                fullWidth
                                id="email"
                                label="Email Address"
                                name="email"
                                autoComplete="email"
                                autoFocus
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                sx={{
                                    "& label.Mui-focused": {
                                        color: "text.secondary",
                                    },
                                    "& .MuiOutlinedInput-root": {
                                        "& fieldset": {
                                            borderColor: "lightgray",
                                        },
                                        "&:hover fieldset": {
                                            borderColor: "text.secondary",
                                        },
                                        "&.Mui-focused fieldset": {
                                            borderColor: "text.secondary",
                                        },
                                    },
                                }}
                            />
                            <TextField
                                margin="normal"
                                required
                                fullWidth
                                name="password"
                                label="Password"
                                type="password"
                                id="password"
                                autoComplete="current-password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                sx={{
                                    "& label.Mui-focused": {
                                        color: "text.secondary",
                                    },
                                    "& .MuiOutlinedInput-root": {
                                        "& fieldset": {
                                            borderColor: "lightgray",
                                        },
                                        "&:hover fieldset": {
                                            borderColor: "text.secondary",
                                        },
                                        "&.Mui-focused fieldset": {
                                            borderColor: "text.secondary",
                                        },
                                    },
                                }}
                            />
                            <Button
                                type="submit"
                                fullWidth
                                variant="contained"
                                sx={{
                                    mt: 3,
                                    mb: 2,
                                    bgcolor: "success.main",
                                    "&:hover": { bgcolor: "secondary.main" },
                                }}
                            >
                                Sign In
                            </Button>
                            <Grid container>
                                <Grid item>
                                    <Link
                                        href="/signup"
                                        variant="body2"
                                        sx={{ color: "text.secondary" }}
                                    >
                                        {"Don't have an account? Sign Up"}
                                    </Link>
                                </Grid>
                            </Grid>
                            <Copyright sx={{ mt: 5 }} />
                        </Box>
                    </Box>
                </Grid>
            </Grid>
        </ThemeProvider>
    );
}
