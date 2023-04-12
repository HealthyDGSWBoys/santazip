<script lang="ts">
	import DeviceReport from "./components/DeviceReport/DeviceReport.svelte";
	import ModelBuilder from "./components/ModelBuilder/ModelBuilder.svelte";

</script>

<main>
	<div class="module-wrapper">
		<ModelBuilder/>
	</div>
	<div class="report-wrapper">
		<DeviceReport/>
	</div>
</main>

<style>
	.module-wrapper {
		width: 100%;
		height: 600px;
	}
	.report-wrapper {
		width: 300px;
		height: 300px;
	}
</style>