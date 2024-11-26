import * as React from "react";
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Link from "@mui/material/Link";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import { ThemeProvider } from "@mui/material/styles";
import { useState } from "react";
import theme from "../styles/theme"; // Asegúrate de importar el mismo tema

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

export default function SignUp() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleSubmit = (event) => {
        event.preventDefault();
        const data = new FormData(event.currentTarget);
        console.log({
            email: data.get("email"),
            password: data.get("password"),
        });
    };

    async function signup() {
        const username = email.split("@")[0];

        try {
            const preresponse = await fetch(
                `https://back-zu3yqmmklq-uc.a.run.app/users/`,
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

            if (!preresponse.ok) {
                throw new Error("Network response was not ok");
            }
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
            <Container component="main" maxWidth="xs">
                <CssBaseline />
                <Box
                    sx={{
                        marginTop: 8,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                    }}
                >
                    <Avatar sx={{ m: 1, bgcolor: "text.secondary" }}>
                        <LockOutlinedIcon />
                    </Avatar>
                    <Typography component="h1" variant="h5">
                        Sign up
                    </Typography>
                    <Box
                        component="form"
                        noValidate
                        onSubmit={handleSubmit}
                        sx={{ mt: 3 }}
                    >
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <TextField
                                    required
                                    fullWidth
                                    id="email"
                                    label="Email Address"
                                    name="email"
                                    autoComplete="email"
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
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    required
                                    fullWidth
                                    name="password"
                                    label="Password"
                                    type="password"
                                    id="password"
                                    onChange={(e) =>
                                        setPassword(e.target.value)
                                    }
                                    autoComplete="new-password"
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
                            </Grid>
                        </Grid>
                        <Button
                            type="submit"
                            fullWidth
                            variant="contained"
                            onClick={signup}
                            sx={{
                                mt: 3,
                                mb: 2,
                                bgcolor: "success.main",
                                "&:hover": { bgcolor: "secondary.main" },
                            }}
                        >
                            Sign Up
                        </Button>
                        <Grid container justifyContent="flex-end">
                            <Grid item>
                                <Link
                                    href="/login"
                                    variant="body2"
                                    sx={{ color: "text.secondary" }}
                                >
                                    Already have an account? Sign in
                                </Link>
                            </Grid>
                        </Grid>
                    </Box>
                </Box>
                <Copyright sx={{ mt: 5 }} />
            </Container>
        </ThemeProvider>
    );
}
