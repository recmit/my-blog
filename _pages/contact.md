---
layout: page
title: Contact
permalink: /contact/
---

<form accept-charset="UTF-8" action="https://formspree.io/f/mjvlaznv" method="POST">
      <div class="form-group">
        <label for="email">Email address</label>
        <input type="email" name="email" class="form-control" id="email" placeholder="Enter email" required>
      </div>
      <div class="form-group">
        <label for="name">Name</label>
        <input type="text" name="name" class="form-control" id="name" placeholder="Enter your name" required>
      </div>
      <!-- <hr> -->
      <div class="form-group">
		<label for="msg">Message</label>
	    <textarea name="message" class="form-control" id="message" placeholder="Your message" required></textarea>
      </div>
      <!-- <hr> -->
      <div class="g-recaptcha" data-sitekey="6Ld23IYfAAAAAPEblWZSCtHZTMr55BRJ2-tCJoiJ"></div>
      <br/>
      <button type="submit" class="btn btn-primary">Submit</button>
</form>