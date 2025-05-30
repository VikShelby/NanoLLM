const passport = require("passport");
const LocalStrategy = require("passport-local").Strategy;
const GoogleStrategy = require("passport-google-oauth20").Strategy;
const bcrypt = require("bcryptjs");
const User = require("../models/userModel");
const { generateTokenAndSetCookie } = require("../utils/helper/generateTokenAndSetCookie"); // JWT function

// Local Strategy for Email/Password Login
module.exports = (passport) => {
passport.use(
  new LocalStrategy({ usernameField: "email" }, async (email, password, done) => {
    try {
      const user = await User.findOne({ email });
      if (!user) return done(null, false, { message: "Invalid username or password" });

      const isPasswordCorrect = await bcrypt.compare(password, user.password);
      if (!isPasswordCorrect) return done(null, false, { message: "Invalid username or password" });

     

      
      return done(null, user);
    } catch (error) {
      return done(error);
    }
  })
);

// Google Strategy for OAuth Login
passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "http://localhost:8080/api/auth/google/callback", // Update with your frontend URL
      passReqToCallback: true,
    },
    async (request, accessToken, refreshToken, profile, done) => {

      try {
        const { name, email, picture } = profile._json;
         console.log(name)
        let user = await User.findOne({ email });
        let isNewUser = false;

        if (!user) {
          // If the user doesn't exist, create a new user
          user = new User({
            name,
            email,
            username: email.split("@")[0], // You can generate username however you'd like
            profilePic: picture,
            svg: "", // You can generate a SVG if needed
          });
          await user.save();
          isNewUser = true;
        }
       
        user.isNewUser = isNewUser;
        return done(null, user);
      } catch (error) {
        return done(error);
      }
    }
  )
);
passport.serializeUser((user, done) => {
    done(null, user._id);
  });

  passport.deserializeUser(async (id, done) => {
    try {
      const user = await User.findById(id);
      done(null, user);
    } catch (err) {
      done(err);
    }
  });
}