const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// Make Metro treat .tflite as a bundled asset
config.resolver.assetExts = [...config.resolver.assetExts, "tflite"];

module.exports = config;
