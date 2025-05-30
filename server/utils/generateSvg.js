const { v4: uuidv4 } = require("uuid");
const ColorHash = require("color-hash").default;

const colorHash = new ColorHash({ saturation: 1.0 });

const stringToColour = (s) => colorHash.hex(s);

const generateColours = (s) => {
  const s1 = s.substring(0, Math.floor(s.length / 2));
  const s2 = s.substring(Math.floor(s.length / 2));
  const c1 = stringToColour(s1);
  const c2 = stringToColour(s2);

  return [c1, c2];
};

const generateSVG = (s, size = 40) => {
  const [c1, c2] = generateColours(s);
  const gradientId = `gradient-${uuidv4()}`; // Unique gradient ID

  const svg = `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="${size}" height="${size}" fill="url(#${gradientId})" />
      <defs>
        <linearGradient id="${gradientId}" x1="0" y1="0" x2="${size}" y2="${size}" gradientUnits="userSpaceOnUse">
          <stop stop-color="${c1}" />
          <stop offset="1" stop-color="${c2}" />
        </linearGradient>
      </defs>
    </svg>
  `.trim();

  return svg;
};

module.exports = generateSVG;
