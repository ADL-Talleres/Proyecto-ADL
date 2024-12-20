import React, { useEffect, useState } from "react";
import { ThemeProvider } from "@mui/material/styles";
import { CssBaseline, Grid } from "@mui/material";
import theme from "../styles/theme";

import {
    Container,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Dialog,
    DialogContent,
    Box
} from "@mui/material";

const YourImages = () => {
    const [tasks, setTasks] = useState([]);
    const [open, setOpen] = useState(false);
    const [modalImage, setModalImage] = useState("");

    const handleOpen = (imageSrc) => {
        setModalImage(imageSrc);
        setOpen(true);
    };

    const handleClose = () => {
        setOpen(false);
    };

    useEffect(() => {
        const fetchTasks = async () => {
            const token = localStorage.getItem("token");
            const email = localStorage.getItem("email");

            try {
                const response = await fetch(
                    `https://back-406206621453.us-central1.run.app/tasks?email=${email}`,
                    {
                        method: "GET",
                        headers: {
                            "Content-Type": "application/json",
                            Authorization: `Bearer ${token}`,
                        },
                    }
                );

                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }

                const data = await response.json();
                setTasks(data);
            } catch (error) {
                console.error("Error fetching tasks:", error);
            }
        };

        fetchTasks();
    }, []);

    return (
        <ThemeProvider theme={theme}>
            <Grid container component="main" sx={{ height: "100vh" }}>
                <CssBaseline />
                <Container>
                    <Typography variant="h4" component="h1" gutterBottom>
                        Your Images
                    </Typography>
                    <TableContainer component={Paper}>
                        <Table
                            sx={{
                                border: "2px solid",
                                borderColor: "text.secondary",
                                backgroundColor: "primary.main",
                            }}
                        >
                            <TableHead>
                                <TableRow>
                                    <TableCell sx={{ color: "text.primary" }}>
                                        Name
                                    </TableCell>
                                    <TableCell sx={{ color: "text.primary" }}>
                                        Timestamp
                                    </TableCell>
                                    <TableCell sx={{ color: "text.primary" }}>
                                        Image
                                    </TableCell>
                                    <TableCell sx={{ color: "text.primary" }}>
                                        Mask
                                    </TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {tasks.map((task) => (
                                    <TableRow key={task.id}>
                                        <TableCell
                                            sx={{ color: "text.secondary" }}
                                        >
                                            {task.name}
                                        </TableCell>
                                        <TableCell
                                            sx={{ color: "text.secondary" }}
                                        >
                                            {new Date(
                                                task.time_stamp
                                            ).toLocaleString()}
                                        </TableCell>
                                        <TableCell sx={{ color: "text.secondary" }}>
                                            <img
                                                src={`https://back-406206621453.us-central1.run.app/uploads/${task.input_path}`}
                                                alt={task.name}
                                                style={{
                                                    width: "100px",
                                                    height: "100px",
                                                    cursor: "pointer",
                                                }}
                                                onClick={() =>
                                                    handleOpen(
                                                        `https://back-406206621453.us-central1.run.app/uploads/${task.input_path}`
                                                    )
                                                }
                                            />
                                        </TableCell>
                                        <TableCell sx={{ color: "text.secondary" }}>
                                            <img
                                                src={`https://back-406206621453.us-central1.run.app/uploads_reason/${task.input_path}`}
                                                alt={task.name}
                                                style={{
                                                    width: "100px",
                                                    height: "100px",
                                                    cursor: "pointer",
                                                }}
                                                onClick={() =>
                                                    handleOpen(
                                                        `https://back-406206621453.us-central1.run.app/uploads_reason/${task.input_path}`
                                                    )
                                                }
                                            />
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Container>
            </Grid>
            <Dialog open={open} onClose={handleClose} maxWidth="lg">
                <DialogContent>
                    <Box
                        component="img"
                        src={modalImage}
                        alt="Expanded view"
                        sx={{
                            maxWidth: "500px",
                            maxHeight: "500px",
                            display: "block",
                            margin: "0 auto",
                        }}
                    />
                </DialogContent>
            </Dialog>
        </ThemeProvider>
    );
};

export default YourImages;
