<template>
  <div ref="interactive_latex_container">
    <jupyter-widget :widget="latex_output" />
  </div>
</template>

<script>
module.exports = {
  mounted() {
    this.jupyter_attachEventHandlers();
  },
  methods: {
    jupyter_attachEventHandlers() {
      let container = this.$refs.interactive_latex_container;
      if (container.innerHTML === "<div></div>") {
        console.log("Not yet!");
        setTimeout(this.jupyter_attachEventHandlers, 100);
      }
      else {
        table = container.querySelector("mjx-itable");

        rows = table.querySelectorAll(":scope > mjx-mtr > mjx-mtd:nth-child(even)");
        for (let i = 0; i < rows.length; i++) {
          rows[i].addEventListener("mouseup", (e) => { this.fire_on_row_clicked(i); });
        }
        console.log("Attached!");
      }
    },
  },
};
</script>
