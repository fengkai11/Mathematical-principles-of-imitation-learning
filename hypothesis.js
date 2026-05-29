window.hypothesisConfig = function () {
  return {
    openSidebar: false
  };
};

const hypothesisScript = document.createElement("script");
hypothesisScript.async = true;
hypothesisScript.src = "https://hypothes.is/embed.js";
document.body.appendChild(hypothesisScript);
