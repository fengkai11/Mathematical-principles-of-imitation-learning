window.hypothesisConfig = function () {
  return {
    openSidebar: true
  };
};

const hypothesisScript = document.createElement("script");
hypothesisScript.async = true;
hypothesisScript.src = "https://hypothes.is/embed.js";
document.body.appendChild(hypothesisScript);
