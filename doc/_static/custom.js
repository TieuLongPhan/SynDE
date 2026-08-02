// Interactive enhancements for SynDE documentation
document.addEventListener("DOMContentLoaded", function () {
  // Add copy button fallback to all code blocks if sphinx_copybutton is missing or for additional compatibility
  document.querySelectorAll("div.highlight pre").forEach(function (block) {
    if (!block.parentElement.querySelector(".copybtn, .synde-copy-btn")) {
      var button = document.createElement("button");
      button.className = "synde-copy-btn";
      button.type = "button";
      button.innerText = "Copy";
      button.setAttribute("aria-label", "Copy code block to clipboard");

      button.addEventListener("click", function () {
        var text = block.innerText;
        navigator.clipboard.writeText(text).then(
          function () {
            button.innerText = "Copied!";
            button.classList.add("copied");
            setTimeout(function () {
              button.innerText = "Copy";
              button.classList.remove("copied");
            }, 2000);
          },
          function (err) {
            console.error("Could not copy text: ", err);
          }
        );
      });

      block.parentElement.style.position = "relative";
      block.parentElement.appendChild(button);
    }
  });

  // Ctrl+K keyboard shortcut to focus search bar
  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      var searchInput = Array.from(document.querySelectorAll("input[name='q']")).find(function (input) {
        return input.offsetParent !== null;
      });
      if (searchInput) {
        searchInput.focus();
        searchInput.select();
      }
    }
  });
});
