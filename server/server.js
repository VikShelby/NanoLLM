// server.js

const express = require('express');
const session = require("express-session");
const connectDB = require('./db/mongodb');
const  dotenv  = require("dotenv")
const cookieParser  = require("cookie-parser")
const userRoutes = require('./routes/userRoutes')
const passport = require("passport");
const cors = require('cors');
const { protectRoute } = require('./middlewares/protectRoute');
dotenv.config();

const app = express();

//Start mongodb
 connectDB();
 
 //middlewares
 const corsOptions = {
    origin: 'http://localhost:3000',
    credentials: true, // Enable credentials (cookies) to be sent and received
};
// Passport middleware initialization
app.use(
  session({
    secret: "your_secret_key", // Replace with a more secure key in production
    resave: false,
    saveUninitialized: false,
    cookie: { httpOnly: true, secure: false }, // Set `secure: true` in production if using HTTPS
  })
);
app.use(passport.initialize());
app.use(passport.session());

// Import and configure Passport strategies
require("./config/passaport")(passport); // Load passport strategies (e.g., local and google)

 app.use(express.json({ limit: "50mb" })); // To parse JSON data in the req.body
 app.use(express.urlencoded({ extended: true })); // To parse form data in the req.body
 app.use(cookieParser());
 
 app.use(cors(corsOptions))
 app.use((req, res, next) => {
  console.log(`Proxy received request:
  - Method: ${req.method} 
  - URL: ${req.originalUrl} 
  - Body: ${JSON.stringify(req.body, null, 2) || 'No body'}`);
  next();
});
 
 app.use("/api/auth", userRoutes);
 app.get('/api/protected', protectRoute, (req, res) => {
  res.json({ message: 'This is a protected route', user: req.user });
});


const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
